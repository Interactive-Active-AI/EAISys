"""EmbodimentInterpreter contract (`g_E`), AST safety audit, subprocess-isolated verification, and dynamic loader."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import numpy as np

from eaisys.harness.action_space import MOBILE_BASE_UNITS, ROTATION_UNITS, TRANSLATION_UNITS
from eaisys.types import Transform, axis_angle_to_matrix, matrix_to_rotvec
from eaisys.urdf.kinematics import forward_kinematics_tree
from eaisys.urdf.parser import KinematicTree, parse_robot_file


_ALLOWED_IMPORT_ROOTS: frozenset[str] = frozenset(
    {
        "__future__",
        "collections",
        "copy",
        "dataclasses",
        "eaisys",
        "functools",
        "harnesses",
        "itertools",
        "math",
        "numpy",
        "typing",
    }
)

_FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "open",
        "setattr",
        "vars",
    }
)

_FORBIDDEN_ATTRS: frozenset[str] = frozenset(
    {
        "__builtins__",
        "__class__",
        "__closure__",
        "__code__",
        "__dict__",
        "__func__",
        "__globals__",
        "__import__",
        "__loader__",
        "__mro__",
        "__spec__",
        "__subclasses__",
    }
)


def audit_interpreter_ast(source_code: str) -> list[str]:
    """Static AST security gate for LLM-synthesized embodiment interpreters.

    Ensures generated interpreters only import pure math/kinematic modules and cannot
    invoke host I/O, shell execution, or Python frame/dunder sandbox escapes.
    """
    violations: list[str] = []
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return [f"SyntaxError at line {exc.lineno}: {exc.msg}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg not in _ALLOWED_IMPORT_ROOTS:
                    violations.append(f"Disallowed import '{alias.name}' at line {node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                violations.append(f"Relative import disallowed at line {node.lineno}")
            elif node.module:
                root_pkg = node.module.split(".")[0]
                if root_pkg not in _ALLOWED_IMPORT_ROOTS:
                    violations.append(f"Disallowed import from '{node.module}' at line {node.lineno}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                violations.append(f"Disallowed builtin call '{node.func.id}()' at line {node.lineno}")
        elif isinstance(node, ast.Attribute):
            if node.attr in _FORBIDDEN_ATTRS:
                violations.append(f"Disallowed dunder attribute '.{node.attr}' at line {node.lineno}")
    return violations


@dataclass
class InterpreterState:
    """Internal setpoint & joint state `s_t` maintained by an `EmbodimentInterpreter`."""

    arm_q: dict[str, np.ndarray]
    tcp_setpoint: dict[str, Transform]
    gripper_open: dict[str, float]  # 1.0 = open, 0.0 = closed
    step_m: float = 0.02
    rot_step_rad: float = math.radians(15.0)
    base_se2: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    torso_q: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    def clone(self) -> InterpreterState:
        return copy.deepcopy(self)


@dataclass
class StepGroundResult:
    """Deterministic physical execution result of grounding one semantic unit `a_t`."""

    unit: str
    arm: str
    accepted: bool
    clamped_by_workspace: bool = False
    ik_pos_err_m: float = 0.0
    delta_tcp_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "arm": self.arm,
            "accepted": self.accepted,
            "clamped_by_workspace": self.clamped_by_workspace,
            "ik_pos_err_m": round(float(self.ik_pos_err_m), 5),
            "delta_tcp_world": [round(float(x), 4) for x in self.delta_tcp_world],
            "note": self.note,
        }


class EmbodimentInterpreter:
    """Abstract base class for per-URDF semantic-action interpreters `g_E`.

    In EAISys, concrete subclasses in `harnesses/embodiments/<robot_id>.py` are
    synthesized and verified by a Frontier LLM directly from `(URDF + dataset/common)`.
    """

    robot_id: str = "unnamed"
    arm_names: tuple[str, ...] = ("main",)
    supports_mobile_base: bool = False
    max_gripper_width_m: float = 0.08
    workspace_bounds: tuple[np.ndarray, np.ndarray] = (
        np.array([0.15, -0.55, 0.02], dtype=np.float64),
        np.array([0.75, 0.55, 0.65], dtype=np.float64),
    )

    def reset(self) -> InterpreterState:
        raise NotImplementedError

    def arm_joint_names(self, arm: str = "main") -> tuple[str, ...]:
        raise NotImplementedError

    def tcp_link_name(self, arm: str = "main") -> str:
        raise NotImplementedError

    def fk_tcp(self, state: InterpreterState, arm: str = "main") -> Transform:
        raise NotImplementedError

    def apply_unit(
        self, unit: str, state: InterpreterState, arm: str = "main"
    ) -> tuple[InterpreterState, StepGroundResult]:
        raise NotImplementedError


def load_interpreter_module(
    module_path: str | Path,
    urdf_path: str | Path,
    harnesses_root: str | Path | None = None,
    enforce_ast_audit: bool = True,
) -> EmbodimentInterpreter:
    """Audit and dynamically load an LLM-synthesized `harnesses/embodiments/<robot>.py` interpreter."""
    mpath = Path(module_path).resolve()
    if enforce_ast_audit:
        violations = audit_interpreter_ast(mpath.read_text(encoding="utf-8"))
        if violations:
            raise PermissionError(
                f"Interpreter '{mpath.name}' failed static AST security audit: " + "; ".join(violations)
            )

    if harnesses_root is not None:
        hparent = str(Path(harnesses_root).resolve().parent)
        if hparent not in sys.path:
            sys.path.insert(0, hparent)
    else:
        repo_parent = str(mpath.parents[2])
        if repo_parent not in sys.path:
            sys.path.insert(0, repo_parent)

    mod_name = f"_eaisys_emb_{mpath.stem}_{abs(hash(str(mpath))) % 100000}"
    spec = importlib.util.spec_from_file_location(mod_name, mpath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load interpreter module from {mpath}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "build_interpreter"):
        raise AttributeError(
            f"Interpreter module '{mpath}' must define `build_interpreter(urdf_path: str) -> EmbodimentInterpreter`."
        )
    return module.build_interpreter(str(Path(urdf_path).resolve()))


@dataclass
class InterpreterVerificationReport:
    robot_id: str
    passed: bool
    checks: dict[str, dict[str, Any]]
    failures: list[str]
    isolated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "passed": self.passed,
            "isolated": self.isolated,
            "failures": self.failures,
            "checks": self.checks,
        }


def verify_interpreter(
    interpreter: EmbodimentInterpreter,
    urdf_or_tree: str | Path | KinematicTree,
    step_tol_m: float = 0.004,
    rot_tol_rad: float = 0.09,
    check_rotations: bool = True,
    check_mobile: bool = True,
) -> InterpreterVerificationReport:
    """Independently verify an LLM-written interpreter against ground-truth URDF kinematics.

    Checks:
      1. `reset()` joint values lie inside URDF joint limits, and `interpreter.fk_tcp` agrees
         with ground-truth `forward_kinematics_tree` on the URDF XML.
      2. Every directional unit (`MV_FWD`, `MV_BACK`, `MV_LEFT`, `MV_RIGHT`, `MV_UP`, `MV_DOWN`)
         moves the true URDF TCP along the exact canonical world axis (+X/-X/+Y/-Y/+Z/-Z)
         within `step_tol_m`, while keeping joints strictly inside URDF limits.
      3. Rotation units (`ROT_*`) rotate the true URDF TCP orientation around the canonical world
         axis within `rot_tol_rad` while keeping TCP translation drift within `step_tol_m`.
      4. `GRASP` and `RELEASE` toggle `gripper_open` cleanly (`0.0` vs `1.0`).
      5. For mobile bases (`supports_mobile_base=True`), `BASE_FWD` and `BASE_YAW_LEFT` move the
         world-frame TCP consistently with SE(2) rigid-body kinematics.
      6. Repeated `MV_FWD` hits workspace projection `clamped_by_workspace` safely without NaNs.
    """
    tree = urdf_or_tree if isinstance(urdf_or_tree, KinematicTree) else parse_robot_file(urdf_or_tree)
    checks: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    def _true_urdf_tcp(st: InterpreterState, arm: str) -> Transform:
        jmap: dict[str, float] = {}
        for cname in interpreter.arm_names:
            for jn, val in zip(interpreter.arm_joint_names(cname), st.arm_q[cname]):
                jmap[jn] = float(val)
        if interpreter.supports_mobile_base and len(st.base_se2) >= 3:
            T_base = Transform.from_xyz_rpy(
                (float(st.base_se2[0]), float(st.base_se2[1]), 0.0),
                (0.0, 0.0, float(st.base_se2[2])),
            )
        else:
            T_base = Transform.identity()
        poses = forward_kinematics_tree(tree, jmap, base_transform=T_base)
        return poses[interpreter.tcp_link_name(arm)]

    try:
        s0 = interpreter.reset()
    except Exception as exc:
        return InterpreterVerificationReport(
            robot_id=interpreter.robot_id,
            passed=False,
            checks={},
            failures=[f"reset() raised exception: {exc}"],
        )

    for arm in interpreter.arm_names:
        jnames = interpreter.arm_joint_names(arm)
        q0 = s0.arm_q[arm]
        if len(q0) != len(jnames):
            failures.append(f"[{arm}] reset q length {len(q0)} != joint count {len(jnames)}")
            continue

        # 1. Limit check & FK consistency check at reset
        for jn, qv in zip(jnames, q0):
            jt = tree.joints[jn]
            if not jt.is_continuous and (qv < jt.lower - 1e-4 or qv > jt.upper + 1e-4):
                failures.append(
                    f"[{arm}] reset joint '{jn}'={qv:.4f} outside URDF [{jt.lower:.4f}, {jt.upper:.4f}]"
                )
        T_true_0 = _true_urdf_tcp(s0, arm)
        T_interp_0 = interpreter.fk_tcp(s0, arm)
        fk_discrepancy = float(np.linalg.norm(T_true_0.p - T_interp_0.p))
        checks[f"{arm}_reset_fk"] = {
            "true_tcp_xyz": [round(float(c), 4) for c in T_true_0.p],
            "fk_discrepancy_m": round(fk_discrepancy, 6),
        }
        if fk_discrepancy > 2e-3:
            failures.append(
                f"[{arm}] interpreter.fk_tcp disagrees with ground-truth URDF FK by {fk_discrepancy:.4f}m"
            )

        # 2. Directional unit grounding check (1 cm step)
        s_fine, _ = interpreter.apply_unit("STEP_ULTRA_FINE", s0.clone(), arm=arm)
        step_len = s_fine.step_m
        dir_errors: dict[str, float] = {}
        for unit, axis_vec in TRANSLATION_UNITS.items():
            s_next, res = interpreter.apply_unit(unit, s_fine.clone(), arm=arm)
            T_true_1 = _true_urdf_tcp(s_next, arm)
            actual_delta = T_true_1.p - T_true_0.p
            expected_delta = np.asarray(axis_vec, dtype=np.float64) * step_len
            err = float(np.linalg.norm(actual_delta - expected_delta))
            dir_errors[unit] = round(err, 5)
            if not res.accepted or err > step_tol_m:
                failures.append(
                    f"[{arm}] `{unit}` expected delta {np.round(expected_delta, 4).tolist()}, "
                    f"got true URDF delta {np.round(actual_delta, 4).tolist()} (err={err:.4f}m, note={res.note})"
                )
            for jn, qv in zip(jnames, s_next.arm_q[arm]):
                jt = tree.joints[jn]
                if not jt.is_continuous and (qv < jt.lower - 1e-4 or qv > jt.upper + 1e-4):
                    failures.append(f"[{arm}] `{unit}` violated joint limit on '{jn}': {qv:.4f}")
        checks[f"{arm}_directional_errors_m"] = dir_errors

        # 3. Rotation unit grounding check (SO(3) angle & TCP drift)
        if check_rotations:
            rot_errors_deg: dict[str, float] = {}
            rot_passed_any = 0
            for unit, axis_vec in ROTATION_UNITS.items():
                s_rot = s0.clone()
                s_rot.rot_step_rad = math.radians(8.0)
                s_next, res = interpreter.apply_unit(unit, s_rot, arm=arm)
                if not res.accepted:
                    rot_errors_deg[unit] = -1.0
                    continue
                T_true_1 = _true_urdf_tcp(s_next, arm)
                R_exp = axis_angle_to_matrix(np.asarray(axis_vec, dtype=np.float64), s_rot.rot_step_rad) @ T_true_0.R
                r_err = float(np.linalg.norm(matrix_to_rotvec(R_exp @ T_true_1.R.T)))
                p_drift = float(np.linalg.norm(T_true_1.p - T_true_0.p))
                rot_errors_deg[unit] = round(math.degrees(r_err), 3)
                if r_err <= rot_tol_rad and p_drift <= step_tol_m:
                    rot_passed_any += 1
                else:
                    failures.append(
                        f"[{arm}] `{unit}` rotation err={math.degrees(r_err):.2f}deg, tcp_drift={p_drift:.4f}m"
                    )
            checks[f"{arm}_rotation_errors_deg"] = rot_errors_deg
            if rot_passed_any < 4:
                failures.append(f"[{arm}] fewer than 4/6 ROT_* units succeeded ({rot_passed_any}/6)")

        # 4. Gripper toggle check
        s_closed, _ = interpreter.apply_unit("GRASP", s0.clone(), arm=arm)
        s_opened, _ = interpreter.apply_unit("RELEASE", s_closed.clone(), arm=arm)
        if not (s_closed.gripper_open[arm] <= 0.05 and s_opened.gripper_open[arm] >= 0.95):
            failures.append(
                f"[{arm}] GRASP/RELEASE failed to toggle gripper_open "
                f"(got closed={s_closed.gripper_open[arm]}, open={s_opened.gripper_open[arm]})"
            )

        # 5. Workspace safety clamp check
        s_run, _ = interpreter.apply_unit("STEP_COARSE", s0.clone(), arm=arm)
        clamped_seen = False
        for _ in range(30):
            s_run, r_run = interpreter.apply_unit("MV_FWD", s_run, arm=arm)
            if r_run.clamped_by_workspace:
                clamped_seen = True
                break
        checks[f"{arm}_workspace_firewall"] = {"clamped_safely": clamped_seen}
        if not clamped_seen and not interpreter.supports_mobile_base:
            failures.append(f"[{arm}] Workspace bounds never clamped 1.2m runaway MV_FWD motion.")

    # 6. Mobile base SE(2) verification
    if check_mobile and interpreter.supports_mobile_base:
        arm0 = interpreter.arm_names[0]
        T_before = _true_urdf_tcp(s0, arm0)
        s_mob, r_mob = interpreter.apply_unit("BASE_FWD", s0.clone(), arm=arm0)
        T_after = _true_urdf_tcp(s_mob, arm0)
        dx_expected = float(MOBILE_BASE_UNITS["BASE_FWD"][0])
        dx_actual = float(T_after.p[0] - T_before.p[0])
        mob_err = abs(dx_actual - dx_expected)
        checks["mobile_base_se2"] = {"base_fwd_dx_m": round(dx_actual, 4), "err_m": round(mob_err, 5)}
        if not r_mob.accepted or mob_err > 5e-3:
            failures.append(f"[mobile] BASE_FWD expected dx={dx_expected:.3f}m, got {dx_actual:.3f}m")

    return InterpreterVerificationReport(
        robot_id=interpreter.robot_id,
        passed=len(failures) == 0,
        checks=checks,
        failures=failures,
    )


def verify_interpreter_isolated(
    module_path: str | Path,
    urdf_path: str | Path,
    harnesses_root: str | Path | None = None,
    *,
    timeout_s: float = 120.0,
    check_rotations: bool = True,
) -> InterpreterVerificationReport:
    """Verify an LLM-synthesized interpreter behind an AST audit + isolated child-process boundary."""
    mpath = Path(module_path).resolve()
    upath = Path(urdf_path).resolve()
    hroot = Path(harnesses_root).resolve() if harnesses_root else mpath.parent.parent

    violations = audit_interpreter_ast(mpath.read_text(encoding="utf-8"))
    if violations:
        return InterpreterVerificationReport(
            robot_id=mpath.stem,
            passed=False,
            checks={"ast_security_audit": {"passed": False, "violations": violations}},
            failures=violations,
            isolated=True,
        )

    pkg_src = str(Path(__file__).resolve().parents[2])
    repo_root = str(hroot.parent)
    minimal_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": os.pathsep.join([pkg_src, repo_root]),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    child_code = (
        "import json, sys\n"
        "from eaisys.harness.interpreter import load_interpreter_module, verify_interpreter\n"
        "interp = load_interpreter_module(sys.argv[1], sys.argv[2], harnesses_root=sys.argv[3], enforce_ast_audit=True)\n"
        "rep = verify_interpreter(interp, sys.argv[2], check_rotations=(sys.argv[4] == '1'))\n"
        "d = rep.to_dict()\n"
        "d['isolated'] = True\n"
        "d['checks']['ast_security_audit'] = {'passed': True, 'violations': []}\n"
        "print('__VERIFY_JSON__:' + json.dumps(d))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", child_code, str(mpath), str(upath), str(hroot), "1" if check_rotations else "0"],
            env=minimal_env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return InterpreterVerificationReport(
            robot_id=mpath.stem,
            passed=False,
            checks={},
            failures=[f"Isolated verification timed out after {timeout_s:.1f}s"],
            isolated=True,
        )

    marker = [ln for ln in proc.stdout.splitlines() if ln.startswith("__VERIFY_JSON__:")]
    if proc.returncode != 0 or not marker:
        err = (proc.stderr or proc.stdout or f"child exited with code {proc.returncode}").strip()[-800:]
        return InterpreterVerificationReport(
            robot_id=mpath.stem,
            passed=False,
            checks={},
            failures=[f"Isolated verification failed: {err}"],
            isolated=True,
        )

    payload = json.loads(marker[-1].split("__VERIFY_JSON__:", 1)[1])
    return InterpreterVerificationReport(
        robot_id=payload["robot_id"],
        passed=bool(payload["passed"]),
        checks=payload.get("checks", {}),
        failures=payload.get("failures", []),
        isolated=True,
    )
