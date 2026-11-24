"""Gymnasium-compatible discrete semantic-action wrapper around EAISys environments (CaP-Gym style)."""

from __future__ import annotations

from typing import Any
import numpy as np

from eaisys.harness.action_space import valid_units_for_embodiment


class EAISysGymEnv:
    """Standard `reset() -> (obs, info)` / `step(action) -> (obs, reward, terminated, truncated, info)` interface.

    Works standalone without requiring `gymnasium`, and exposes `action_units` so RL / agentic
    benchmarks can step ANY LLM-synthesized URDF interpreter through the same discrete semantic space.
    """

    def __init__(self, inner_env: Any, max_steps: int = 80) -> None:
        self.inner_env = inner_env
        self.max_steps = int(max_steps)
        interp = getattr(inner_env, "interpreter", None)
        supports_mobile = bool(getattr(interp, "supports_mobile_base", False))
        self.action_units: list[str] = valid_units_for_embodiment(supports_mobile_base=supports_mobile)
        self._steps = 0

    @property
    def num_actions(self) -> int:
        return len(self.action_units)

    def _observe(self) -> dict[str, Any]:
        if hasattr(self.inner_env, "compose_frame_rgb"):
            rgb = self.inner_env.compose_frame_rgb()
            tcp = self.inner_env.proprio_tcp_in_base()
            g_cmd, g_qpos = self.inner_env.raw_gripper_proprio()
            return {
                "rgb": rgb,
                "tcp_base_xyz": np.asarray(tcp, dtype=np.float64),
                "gripper_cmd": g_cmd,
                "gripper_qpos": np.asarray(g_qpos, dtype=np.float64),
                "step_m": float(self.inner_env.state.step_m),
            }
        # Fallback for lightweight TabletopTaskEnv
        arm = self.inner_env.interpreter.arm_names[0]
        tcp = self.inner_env.interpreter.fk_tcp(self.inner_env.state, arm=arm).p
        return {
            "tcp_base_xyz": np.asarray(tcp, dtype=np.float64),
            "gripper_open": float(self.inner_env.state.gripper_open.get(arm, 1.0)),
            "step_m": float(self.inner_env.state.step_m),
        }

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None and hasattr(self.inner_env, "seed"):
            self.inner_env.seed = int(seed)
        if hasattr(self.inner_env, "reset"):
            self.inner_env.reset()
        self._steps = 0
        return self._observe(), {"action_units": list(self.action_units), "zero_privilege": True}

    def step(self, action: int | str) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        unit = self.action_units[int(action)] if isinstance(action, (int, np.integer)) else str(action)
        if unit not in self.action_units:
            raise ValueError(f"Unknown semantic action unit: {unit!r}")

        self._steps += 1
        step_info: dict[str, Any]
        if unit == "DONE":
            step_info = {"unit": "DONE", "accepted": True}
        elif hasattr(self.inner_env, "compose_frame_rgb"):
            step_info = self.inner_env.step_unit(unit)
        else:
            arm = self.inner_env.interpreter.arm_names[0]
            step_info = self.inner_env.step_unit(unit, arm=arm).to_dict()

        if hasattr(self.inner_env, "check_success_audit_only"):
            ok, metrics = self.inner_env.check_success_audit_only()
        else:
            ok, metrics = self.inner_env.check_success()

        terminated = bool(ok) or unit == "DONE"
        truncated = (not terminated) and (self._steps >= self.max_steps)
        reward = 1.0 if ok else 0.0
        return self._observe(), reward, terminated, truncated, {"step": step_info, "audit": metrics}

    def close(self) -> None:
        if hasattr(self.inner_env, "close"):
            self.inner_env.close()
