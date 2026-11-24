"""Frontier-LLM synthesis & refactoring loop for URDF-specific and cross-URDF harnesses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np

from eaisys.backends.base import FrontierBackend
from eaisys.datasets.schema import RawEpisode
from eaisys.harness.interpreter import (
    InterpreterVerificationReport,
    audit_interpreter_ast,
    load_interpreter_module,
    verify_interpreter,
    verify_interpreter_isolated,
)


_SUBSTRATE_API_CHEATSHEET = """\
## Exact Substrate API Reference (Do NOT guess signatures — use these exact types & methods)

```python
from eaisys.types import Transform, axis_angle_to_matrix, rpy_to_matrix
from eaisys.urdf import IKResult, KinematicChainSolver, KinematicTree, forward_kinematics_tree, parse_robot_file
from eaisys.harness.action_space import MOBILE_BASE_UNITS, ROTATION_UNITS, STEP_MODE_UNITS, TRANSLATION_UNITS
from eaisys.harness.interpreter import EmbodimentInterpreter, InterpreterState, StepGroundResult

# 1. Transform (frozen dataclass):
#    fields: R: np.ndarray (3, 3), p: np.ndarray (3,)
#    Transform.identity() -> Transform
#    Transform.from_xyz_rpy(xyz=(0,0,0), rpy=(0,0,0)) -> Transform
#    T1.compose(T2) -> Transform

# 2. parse_robot_file(path: str | Path) -> KinematicTree
#    tree.root_link: str
#    tree.joints: dict[str, JointNode]
#      joint.name: str, joint.joint_type: str ("revolute"|"continuous"|"prismatic"|"fixed")
#      joint.parent: str, joint.child: str, joint.lower: float, joint.upper: float, joint.is_continuous: bool
#    forward_kinematics_tree(tree: KinematicTree, joint_values: dict[str, float], base_transform: Transform | None = None) -> dict[str, Transform]

# 3. KinematicChainSolver:
#    solver = KinematicChainSolver(
#        tree=tree,                              # KinematicTree
#        root_link=root_link,                    # str (e.g., "panda_link0" or "left_base_link")
#        tcp_link=tcp_link,                      # str (e.g., "panda_hand_tcp" or "left_ee_tcp")
#        actuated_joint_names=joint_names_tuple, # tuple[str, ...]
#        lower_limits=lower_np_array,            # np.ndarray shape (D,)
#        upper_limits=upper_np_array,            # np.ndarray shape (D,)
#        continuous_mask=cont_bool_np_array,     # np.ndarray shape (D,) dtype=bool
#        home_q=home_q_np_array,                 # np.ndarray shape (D,)
#    )
#    solver.fk(q: np.ndarray, T_root: Transform | None = None) -> Transform
#    solver.manipulability(q: np.ndarray, T_root: Transform | None = None) -> float
#    solver.clamp_or_wrap(q: np.ndarray) -> np.ndarray
#    ik_res: IKResult = solver.solve_ik(
#        target_world=target_tf,   # Transform in world frame!
#        q_seed=q_current,         # np.ndarray shape (D,)
#        T_root=T_chain_root,      # Transform of `root_link` in world frame! (CRITICAL on bimanual/mobile robots)
#        pos_tol_m=1.5e-3,
#        rot_tol_rad=2.5e-2,
#        rot_weight=1.0,           # On 6-DoF arms, fall back to rot_weight=0.15 if rot_weight=1.0 fails
#        max_iters=120,
#        num_restarts=8,
#    )
#    # IKResult fields: success: bool, q: np.ndarray, pos_err_m: float, rot_err_rad: float

# 4. Action dictionaries:
#    TRANSLATION_UNITS = {"MV_FWD": (1,0,0), "MV_BACK": (-1,0,0), "MV_LEFT": (0,1,0), "MV_RIGHT": (0,-1,0), "MV_UP": (0,0,1), "MV_DOWN": (0,0,-1)}
#    ROTATION_UNITS = {"ROT_ROLL_CW": (1,0,0), "ROT_ROLL_CCW": (-1,0,0), "ROT_PITCH_CW": (0,1,0), "ROT_PITCH_CCW": (0,-1,0), "ROT_YAW_CW": (0,0,-1), "ROT_YAW_CCW": (0,0,1)}
#    STEP_MODE_UNITS = {"STEP_COARSE": 0.04, "STEP_FINE": 0.02, "STEP_ULTRA_FINE": 0.01}
#    MOBILE_BASE_UNITS = {"BASE_FWD": (0.15,0,0), "BASE_BACK": (-0.15,0,0), "BASE_LEFT": (0,0.15,0), "BASE_RIGHT": (0,-0.15,0), "BASE_YAW_LEFT": (0,0,0.2618), "BASE_YAW_RIGHT": (0,0,-0.2618)}

# 5. EmbodimentInterpreter abstract methods & return types (MUST implement ALL FIVE!):
#    class MyInterpreter(EmbodimentInterpreter):
#        robot_id = "..."
#        arm_names = ("main",)  # or ("left", "right")
#        supports_mobile_base = False
#        max_gripper_width_m = 0.08
#        workspace_bounds = (np.array([...]), np.array([...]))
#
#        def arm_joint_names(self, arm: str = "main") -> tuple[str, ...]: ...
#        def tcp_link_name(self, arm: str = "main") -> str: ...
#        def reset(self) -> InterpreterState: ...
#        def fk_tcp(self, state: InterpreterState, arm: str = "main") -> Transform: ...
#        def apply_unit(self, unit: str, state: InterpreterState, arm: str = "main") -> tuple[InterpreterState, StepGroundResult]:
#            # MUST return a 2-tuple `(state, StepGroundResult(...))`!
#
#    InterpreterState(arm_q: dict[str, np.ndarray], tcp_setpoint: dict[str, Transform], gripper_open: dict[str, float],
#                     step_m: float = 0.02, rot_step_rad: float = 0.261799, base_se2: np.ndarray = zeros(3), torso_q: np.ndarray = zeros(0))
#    StepGroundResult(unit: str, arm: str, accepted: bool, clamped_by_workspace: bool = False,
#                     ik_pos_err_m: float = 0.0, delta_tcp_world: np.ndarray = zeros(3), note: str = "")
```
"""


def summarize_dataset_episodes(episodes: list[RawEpisode] | None) -> str:
    if not episodes:
        return "ZERO demonstrations provided (strict Zero-Shot Unseen URDF onboarding)."
    rows = []
    for idx, ep in enumerate(episodes[:3], 1):
        arm_desc = []
        for cname, st in ep.arms.items():
            q_info = (
                f"qpos shape={st.qpos.shape}, q0={np.round(st.qpos[0], 3).tolist()}, "
                f"q_min={np.round(st.qpos.min(0), 3).tolist()}, q_max={np.round(st.qpos.max(0), 3).tolist()}"
                if st.qpos is not None
                else "qpos=None"
            )
            e_info = (
                f"eef0={np.round(st.eef_pose7_xyzw[0, :3], 3).tolist()}"
                if st.eef_pose7_xyzw is not None
                else "eef=None"
            )
            arm_desc.append(f"arm `{cname}`: {q_info}, {e_info}")
        rows.append(
            f"- Demo {idx}: dataset=`{ep.dataset_name}`, task=\"{ep.task_instruction}\", "
            f"fps={ep.fps}, " + "; ".join(arm_desc)
        )
    return "\n".join(rows)


@dataclass
class SynthesisTurnRecord:
    attempt: int
    tokens_used: int
    elapsed_s: float
    passed: bool
    failures: list[str]
    prompt: str = ""
    response_text: str = ""
    token_breakdown: dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisReport:
    robot_id: str
    model: str
    urdf_path: str
    output_module_path: str
    used_common_library: bool
    num_demos_provided: int
    passed: bool
    attempts: int
    total_tokens: int
    total_elapsed_s: float
    verification: dict[str, Any]
    turns: list[SynthesisTurnRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "model": self.model,
            "urdf_path": self.urdf_path,
            "output_module_path": self.output_module_path,
            "used_common_library": self.used_common_library,
            "num_demos_provided": self.num_demos_provided,
            "passed": self.passed,
            "attempts": self.attempts,
            "total_tokens": self.total_tokens,
            "total_elapsed_s": round(self.total_elapsed_s, 3),
            "verification": self.verification,
            "turns": [
                {
                    "attempt": t.attempt,
                    "tokens_used": t.tokens_used,
                    "elapsed_s": t.elapsed_s,
                    "passed": t.passed,
                    "failures": t.failures,
                    **(
                        {
                            "token_breakdown": t.token_breakdown,
                            "prompt": t.prompt,
                            "response_text": t.response_text,
                        }
                        if t.prompt
                        else {}
                    ),
                }
                for t in self.turns
            ],
        }


class URDFHarnessSynthesizer:
    """Uses a Frontier LLM to write, verify, and refactor URDF semantic-action interpreters."""

    def __init__(self, backend: FrontierBackend, max_attempts: int = 3) -> None:
        self.backend = backend
        self.max_attempts = max_attempts

    def synthesize_embodiment_interpreter(
        self,
        urdf_path: str | Path,
        output_py_path: str | Path,
        sidecar_path: str | Path | None = None,
        sample_episodes: list[RawEpisode] | None = None,
        common_library_path: str | Path | None = None,
        record_full_trace: bool = False,
    ) -> SynthesisReport:
        upath = Path(urdf_path).resolve()
        opath = Path(output_py_path).resolve()
        opath.parent.mkdir(parents=True, exist_ok=True)

        urdf_xml = upath.read_text(encoding="utf-8")
        sidecar_txt = (
            Path(sidecar_path).read_text(encoding="utf-8")
            if sidecar_path and Path(sidecar_path).is_file()
            else "None (raw URDF only — infer joints, TCP link, gripper width, and non-singular home_q directly from the URDF XML)."
        )
        common_code = (
            Path(common_library_path).read_text(encoding="utf-8")
            if common_library_path and Path(common_library_path).is_file()
            else ""
        )
        demos_summary = summarize_dataset_episodes(sample_episodes)

        turns: list[SynthesisTurnRecord] = []
        last_code = ""
        last_failures: list[str] = []
        total_tokens = 0
        total_elapsed = 0.0
        final_ver: InterpreterVerificationReport | None = None

        for attempt in range(1, self.max_attempts + 1):
            prompt = self._build_interpreter_prompt(
                urdf_xml=urdf_xml,
                sidecar_txt=sidecar_txt,
                demos_summary=demos_summary,
                common_code=common_code,
                last_code=last_code,
                last_failures=last_failures,
            )
            resp = self.backend.generate(prompt, timeout_s=900.0)
            total_tokens += resp.tokens_used
            total_elapsed += resp.elapsed_s
            code = resp.extract_python_code()
            opath.write_text(code + "\n", encoding="utf-8")

            try:
                harnesses_root = opath.parent.parent
                ast_issues = audit_interpreter_ast(code)
                if ast_issues:
                    ver = InterpreterVerificationReport(
                        robot_id=upath.parent.name,
                        passed=False,
                        checks={"ast_security_audit": {"passed": False, "violations": ast_issues}},
                        failures=ast_issues,
                        isolated=True,
                    )
                else:
                    ver = verify_interpreter_isolated(
                        opath, upath, harnesses_root=harnesses_root, check_rotations=True
                    )
                final_ver = ver
                from eaisys.sanitize import sanitize_payload, sanitize_text

                turns.append(
                    SynthesisTurnRecord(
                        attempt=attempt,
                        tokens_used=resp.tokens_used,
                        elapsed_s=resp.elapsed_s,
                        passed=ver.passed,
                        failures=[sanitize_text(x) for x in ver.failures],
                        prompt=sanitize_text(prompt) if record_full_trace else "",
                        response_text=sanitize_text(resp.text) if record_full_trace else "",
                        token_breakdown=sanitize_payload(resp.raw_meta) if record_full_trace else {},
                    )
                )
                if ver.passed:
                    interp = load_interpreter_module(opath, upath, harnesses_root=harnesses_root)
                    return SynthesisReport(
                        robot_id=interp.robot_id,
                        model=self.backend.model_name,
                        urdf_path=upath.name,
                        output_module_path=str(opath.relative_to(opath.parents[2])),
                        used_common_library=bool(common_code),
                        num_demos_provided=len(sample_episodes or ()),
                        passed=True,
                        attempts=attempt,
                        total_tokens=total_tokens,
                        total_elapsed_s=total_elapsed,
                        verification=ver.to_dict(),
                        turns=turns,
                    )
                last_code = code
                last_failures = list(ver.failures)
            except Exception as exc:
                import traceback

                from eaisys.sanitize import sanitize_payload, sanitize_text

                tb_last = traceback.format_exc().strip().splitlines()[-3:]
                err_msg = sanitize_text(f"{type(exc).__name__}: {exc} ({' | '.join(tb_last)})")
                turns.append(
                    SynthesisTurnRecord(
                        attempt=attempt,
                        tokens_used=resp.tokens_used,
                        elapsed_s=resp.elapsed_s,
                        passed=False,
                        failures=[err_msg],
                        prompt=sanitize_text(prompt) if record_full_trace else "",
                        response_text=sanitize_text(resp.text) if record_full_trace else "",
                        token_breakdown=sanitize_payload(resp.raw_meta) if record_full_trace else {},
                    )
                )
                last_code = code
                last_failures = [err_msg]

        return SynthesisReport(
            robot_id=upath.parent.name,
            model=self.backend.model_name,
            urdf_path=upath.name,
            output_module_path=str(opath.relative_to(opath.parents[2])),
            used_common_library=bool(common_code),
            num_demos_provided=len(sample_episodes or ()),
            passed=False,
            attempts=len(turns),
            total_tokens=total_tokens,
            total_elapsed_s=total_elapsed,
            verification=final_ver.to_dict() if final_ver else {"passed": False, "failures": last_failures},
            turns=turns,
        )

    def synthesize_common_library(
        self,
        seed_interpreter_paths: list[str | Path],
        output_common_path: str | Path,
    ) -> dict[str, Any]:
        """Ask the Frontier LLM to inspect multiple per-URDF interpreters and abstract `harnesses/common/urdf_interpreter_base.py`."""
        snippets = []
        for p in seed_interpreter_paths:
            pp = Path(p)
            snippets.append(f"### `{pp.name}`\n```python\n{pp.read_text(encoding='utf-8')}\n```")

        joined = "\n\n".join(snippets)
        prompt = f"""You are the core architect of EAISys, an LLM-centric URDF Embodied Harness.
Below are verified per-URDF semantic-action interpreters that you previously wrote for distinct robot morphologies:

{joined}

{_SUBSTRATE_API_CHEATSHEET}

## Task: Abstract the Cross-URDF Reusable Common Harness Library
Factor out the morphology-invariant logic shared across single-arm (6-DoF & 7-DoF, revolute & continuous joints), bimanual, and mobile-base URDFs into a concise, reusable Python module `harnesses/common/urdf_interpreter_base.py` (~160-240 lines).

Provide:
- `ChainConfig` dataclass (`name`, `root_link`, `tcp_link`, `joint_names`, `home_q: list[float] | None = None`)
- `GenericURDFInterpreter(EmbodimentInterpreter)` accepting `(urdf_path, robot_id, chains: list[ChainConfig], supports_mobile_base=False, torso_joints=(), max_gripper_width_m=0.08, workspace_bounds=None)`:
  1. If `home_q` is `None`, automatically sample ~120 random configurations inside `[-2.0, 2.0]` (intersected with joint limits) to pick a non-singular bent-elbow ready posture with high `solver.manipulability(q)` and TCP in front of root (`dx > 0.28m, dz > 0.14m`).
  2. If `workspace_bounds` is `None`, automatically set `(min_tcp - [0.22, 0.35, 0.18], max_tcp + [0.28, 0.35, 0.25])` around the `reset()` TCP(s) so all 6 directional steps have plenty of margin while +1.2m runaway `MV_FWD` clamps cleanly.
  3. Implement `reset()`, `arm_joint_names()`, `tcp_link_name()`, `fk_tcp()`, and `apply_unit()` cleanly so that any new seen or unseen URDF interpreter in `harnesses/embodiments/<robot>.py` can be written in ~30-50 lines!

Output ONLY valid Python source code inside a single ```python ... ``` block.
"""
        resp = self.backend.generate(prompt, timeout_s=900.0)
        code = resp.extract_python_code()
        out_p = Path(output_common_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(code + "\n", encoding="utf-8")
        return {
            "model": self.backend.model_name,
            "output_path": str(out_p.relative_to(out_p.parents[2])),
            "tokens_used": resp.tokens_used,
            "elapsed_s": resp.elapsed_s,
            "lines_of_code": len(code.splitlines()),
        }

    @staticmethod
    def _build_interpreter_prompt(
        urdf_xml: str,
        sidecar_txt: str,
        demos_summary: str,
        common_code: str,
        last_code: str,
        last_failures: list[str],
    ) -> str:
        common_section = ""
        if common_code:
            common_section = (
                "\n## Available Cross-URDF Common Library (`harnesses.common.urdf_interpreter_base`)\n"
                "Import and instantiate/subclass from `harnesses.common.urdf_interpreter_base` — "
                "write a concise (~35-60 line) embodiment module specifying this URDF's exact chains, TCP link(s), "
                "gripper width, mobile/torso flags, and non-singular home posture:\n"
                f"```python\n{common_code}\n```\n"
            )

        repair_section = ""
        if last_failures:
            repair_section = (
                "\n## Previous Attempt Failed Ground-Truth URDF Verification\n"
                f"Previous code:\n```python\n{last_code}\n```\n"
                "Verification errors:\n"
                + "\n".join(f"- {e}" for e in last_failures)
                + "\nFix every failure above.\n"
            )

        return f"""You are a Frontier LLM synthesizing a concise, physically grounded semantic-action interpreter `g_E` (~90-150 lines) for a robot from its URDF.

## Target Robot URDF XML
```xml
{urdf_xml}
```

## Optional Sidecar Metadata
```yaml
{sidecar_txt}
```

## Available Dataset Trajectory Statistics
{demos_summary}

{_SUBSTRATE_API_CHEATSHEET}
{common_section}
## Required Contract for `build_interpreter(urdf_path: str) -> EmbodimentInterpreter`
1. Set `robot_id`, `arm_names` (`("main",)` for 1-arm, `("left", "right")` for 2-arm), `supports_mobile_base` (`bool`), `max_gripper_width_m` (`float`), and `workspace_bounds = (ws_lo_np, ws_hi_np)` in world frame so `reset()` TCP has >= `0.10m` margin inside `workspace_bounds` on all 6 axes while +1.2m `MV_FWD` clamps (`clamped_by_workspace=True`) for non-mobile robots.
2. In `reset()`, pick a **non-singular bent-elbow `home_q`** strictly inside URDF limits (if sidecar has `home_q`, verify its manipulability or sample a well-conditioned posture if raw URDF has no sidecar). Set `tcp_setpoint[arm] = self.fk_tcp(state, arm)` and `gripper_open[arm] = 1.0`.
3. Compute chain root transform `T_root = forward_kinematics_tree(self.tree, torso_joint_dict, base_transform=T_base)[root_link]` and pass `T_root=T_root` into `solver.fk(q, T_root=T_root)` and `solver.solve_ik(..., T_root=T_root)` so `fk_tcp` matches ground-truth `forward_kinematics_tree` to `< 1e-6 m`!
4. In `apply_unit(unit, state, arm)`:
   - Handle `STEP_COARSE`/`STEP_FINE`/`STEP_ULTRA_FINE`, `GRASP` (`0.0`), `RELEASE` (`1.0`), `HOLD`/`DONE`.
   - For `MV_*`: target position is `p_raw = state.tcp_setpoint[arm].p + state.step_m * np.array(TRANSLATION_UNITS[unit])`. If not `supports_mobile_base`, clip `p_clamped = np.clip(p_raw, ws_lo, ws_hi)` and set `clamped = not np.allclose(p_raw, p_clamped)`. Solve IK with `rot_weight=1.0`; on 6-DoF arms if `not ik_res.success`, retry with `rot_weight=0.15` and accept if `ik_res.pos_err_m <= 0.003`. Update `state.arm_q[arm] = ik_res.q` and `state.tcp_setpoint[arm] = self.fk_tcp(state, arm)`.
   - For `ROT_*`: `R_new = axis_angle_to_matrix(np.array(ROTATION_UNITS[unit]), state.rot_step_rad) @ state.tcp_setpoint[arm].R`.
   - For `BASE_*` (when `supports_mobile_base=True`): add `(dx, dy, dyaw)` in robot heading frame to `state.base_se2` and refresh `tcp_setpoint`.
{repair_section}
Keep the code clean and direct (do NOT use `inspect` reflection hacks). Output ONLY Python code in a ```python ... ``` block.
"""
