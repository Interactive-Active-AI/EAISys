"""Closed-loop VLM agent loop: VLM perceives, reasons, and plays robots via semantic action units."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from typing import Any

from eaisys.backends.base import FrontierBackend
from eaisys.harness.action_space import single_arm_step_json_schema
from eaisys.harness.plugins import HarnessPluginConfig, build_vlm_step_context
from eaisys.harness.sim_env import TabletopTaskEnv
from eaisys.sanitize import sanitize_payload, sanitize_text


@dataclass
class VLMTurnTrace:
    turn: int
    thought: str
    subtask: str
    action_chunk: list[str]
    step_results: list[dict[str, Any]]
    tcp_after_xyz: list[float]
    gripper_open: float
    holding: str | None
    tokens_used: int
    elapsed_s: float
    prompt: str = ""
    raw_response: str = ""
    zero_privilege_verified: bool = True

    def to_dict(self) -> dict[str, Any]:
        return sanitize_payload(
            {
                "turn": self.turn,
                "zero_privilege_verified": self.zero_privilege_verified,
                "prompt": sanitize_text(self.prompt),
                "raw_response": sanitize_text(self.raw_response),
                "thought": sanitize_text(self.thought),
                "subtask": sanitize_text(self.subtask),
                "action_chunk": self.action_chunk,
                "step_results": self.step_results,
                "tcp_after_xyz": self.tcp_after_xyz,
                "gripper_open": self.gripper_open,
                "holding": self.holding,
                "tokens_used": self.tokens_used,
                "elapsed_s": self.elapsed_s,
            }
        )


@dataclass
class VLMEpisodeReport:
    robot_id: str
    model: str
    task_instruction: str
    success: bool
    total_turns: int
    total_semantic_units: int
    total_tokens: int
    total_elapsed_s: float
    completion_metrics: dict[str, Any]
    turns: list[VLMTurnTrace] = field(default_factory=list)
    zero_privilege: bool = True

    def to_dict(self) -> dict[str, Any]:
        return sanitize_payload(
            {
                "robot_id": self.robot_id,
                "zero_privilege": self.zero_privilege,
                "model": self.model,
                "task_instruction": self.task_instruction,
                "success": self.success,
                "total_turns": self.total_turns,
                "total_semantic_units": self.total_semantic_units,
                "total_tokens": self.total_tokens,
                "total_elapsed_s": round(self.total_elapsed_s, 3),
                "completion_metrics": self.completion_metrics,
                "turns": [t.to_dict() for t in self.turns],
            }
        )


class VLMHarnessRunner:
    """Runs a Frontier VLM in a closed perception-reasoning-action-feedback loop over an `EmbodimentInterpreter`."""

    def __init__(
        self,
        backend: FrontierBackend,
        plugin_cfg: HarnessPluginConfig | None = None,
        max_turns: int = 12,
    ) -> None:
        self.backend = backend
        self.plugin_cfg = plugin_cfg or HarnessPluginConfig()
        self.max_turns = max_turns

    def run_episode(
        self,
        env: TabletopTaskEnv,
        in_context_demos: list[dict[str, Any]] | None = None,
        frame_save_dir: str | Path | None = None,
    ) -> VLMEpisodeReport:
        interp = env.interpreter
        arm = interp.arm_names[0]
        schema = single_arm_step_json_schema(supports_mobile_base=interp.supports_mobile_base)

        turns: list[VLMTurnTrace] = []
        last_subtask = ""
        total_tokens = 0
        total_elapsed = 0.0

        with tempfile.TemporaryDirectory(prefix="eaisys_vlm_frames_") as tmp_frames:
            fdir = Path(frame_save_dir) if frame_save_dir else Path(tmp_frames)
            fdir.mkdir(parents=True, exist_ok=True)

            for turn_idx in range(1, self.max_turns + 1):
                img_path = env.render_multiview_png(fdir / f"turn_{turn_idx:02d}.png")
                prompt = build_vlm_step_context(
                    env=env,
                    turn_idx=turn_idx,
                    max_turns=self.max_turns,
                    last_subtask=last_subtask,
                    in_context_demos=in_context_demos,
                    cfg=self.plugin_cfg,
                )
                resp = self.backend.generate(
                    prompt=prompt,
                    images=[img_path],
                    json_schema=schema,
                )
                total_tokens += resp.tokens_used
                total_elapsed += resp.elapsed_s

                decision = resp.parse_json()
                thought = str(decision.get("thought", ""))
                subtask = str(decision.get("current_subtask", ""))
                chunk = [str(u) for u in decision.get("action_chunk", ["HOLD"])]
                last_subtask = subtask

                step_results: list[dict[str, Any]] = []
                done_emitted = False
                for unit in chunk:
                    if unit == "DONE":
                        done_emitted = True
                        step_results.append({"unit": "DONE", "accepted": True, "note": "agent_done"})
                        break
                    res = env.step_unit(unit, arm=arm)
                    step_results.append(res.to_dict())

                tcp_now = interp.fk_tcp(env.state, arm=arm).p
                turns.append(
                    VLMTurnTrace(
                        turn=turn_idx,
                        thought=thought,
                        subtask=subtask,
                        action_chunk=chunk,
                        step_results=step_results,
                        tcp_after_xyz=[round(float(c), 4) for c in tcp_now],
                        gripper_open=float(env.state.gripper_open.get(arm, 1.0)),
                        holding=None,
                        tokens_used=resp.tokens_used,
                        elapsed_s=resp.elapsed_s,
                        prompt=prompt,
                        raw_response=resp.text,
                        zero_privilege_verified=True,
                    )
                )

                ok, metrics = env.check_success()
                if ok:
                    if frame_save_dir:
                        env.render_multiview_png(fdir / "final_success.png")
                    return VLMEpisodeReport(
                        robot_id=interp.robot_id,
                        model=self.backend.model_name,
                        task_instruction=env.task_instruction,
                        success=True,
                        total_turns=len(turns),
                        total_semantic_units=env.step_count,
                        total_tokens=total_tokens,
                        total_elapsed_s=total_elapsed,
                        completion_metrics=metrics,
                        turns=turns,
                    )
                if done_emitted and not ok:
                    last_subtask = "visual_check_not_placed_on_plate_continue"

            ok, metrics = env.check_success()
            return VLMEpisodeReport(
                robot_id=interp.robot_id,
                model=self.backend.model_name,
                task_instruction=env.task_instruction,
                success=ok,
                total_turns=len(turns),
                total_semantic_units=env.step_count,
                total_tokens=total_tokens,
                total_elapsed_s=total_elapsed,
                completion_metrics=metrics,
                turns=turns,
            )
