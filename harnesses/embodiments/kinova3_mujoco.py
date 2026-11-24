"""URDF-grounded Kinova Gen3 interpreter with a reachable forward boundary."""

import numpy as np

from eaisys.types import Transform, axis_angle_to_matrix
from eaisys.urdf import KinematicChainSolver
from eaisys.harness.action_space import (
    ROTATION_UNITS, STEP_MODE_UNITS, TRANSLATION_UNITS,
)
from eaisys.harness.interpreter import (
    EmbodimentInterpreter, InterpreterState, StepGroundResult,
)
from harnesses.common.urdf_interpreter_base import (
    ChainConfig, GenericURDFInterpreter,
)


class Kinova3Interpreter(GenericURDFInterpreter):
    JOINTS = tuple(f"Actuator{i}" for i in range(1, 8))

    def __init__(self, urdf_path: str):
        super().__init__(
            urdf_path=urdf_path,
            robot_id="kinova3_mujoco",
            chains=[ChainConfig(
                name="main",
                root_link="fixed_base_link",
                tcp_link="kinova3_mujoco_tcp",
                joint_names=self.JOINTS,
                home_q=None,
            )],
            supports_mobile_base=False,
            torso_joints=(),
            max_gripper_width_m=2.0 * 0.0425,
            workspace_bounds=None,
        )
        nodes = [self.tree.joints[name] for name in self.JOINTS]
        lower = np.array([j.lower for j in nodes], dtype=float)
        upper = np.array([j.upper for j in nodes], dtype=float)
        continuous = np.array([j.is_continuous for j in nodes], dtype=bool)
        options = dict(
            tree=self.tree,
            root_link="fixed_base_link",
            tcp_link="kinova3_mujoco_tcp",
            actuated_joint_names=self.JOINTS,
            lower_limits=lower,
            upper_limits=upper,
            continuous_mask=continuous,
        )
        root = self._root(self.reset(), "main")
        rng = np.random.default_rng(7)
        lo = np.where(continuous, -1.8, np.maximum(lower, -1.8))
        hi = np.where(continuous, 1.8, np.minimum(upper, 1.8))
        candidates = [
            self._homes["main"].copy(),
            *rng.uniform(lo, hi, size=(96, len(self.JOINTS))),
        ]

        for home in candidates:
            solver = self.solvers["main"]
            quality = solver.manipulability(home, T_root=root)
            pose = solver.fk(home, T_root=root)
            if (abs(home[3]) < 0.35 or not np.isfinite(quality)
                    or quality <= 1e-6 or not np.all(np.isfinite(pose.p))
                    or pose.p[0] <= 0.28 or pose.p[2] <= 0.14):
                continue
            self._homes["main"] = home.copy()
            self.solvers["main"] = KinematicChainSolver(
                **options, home_q=home.copy(),
            )
            # Keep the forward face close enough to reach before full extension.
            # Every home-to-face margin exceeds 0.10 m.
            self.workspace_bounds = (
                pose.p - np.array([0.20, 0.25, 0.18]),
                pose.p + np.array([0.11, 0.25, 0.22]),
            )
            # Certify actual accepted motion to the boundary for every step mode
            # and for a single long command; an unreachable box is insufficient.
            valid = True
            for step in (*STEP_MODE_UNITS.values(), 1.2):
                probe = self.reset()
                probe.step_m = float(step)
                reached = False
                for _ in range(int(np.ceil(0.16 / step)) + 2):
                    probe, report = self.apply_unit("MV_FWD", probe)
                    if not report.accepted:
                        break
                    if report.clamped_by_workspace:
                        reached = True
                        break
                if not reached:
                    valid = False
                    break
            if valid:
                return
        raise ValueError("No nonsingular home with a reachable forward boundary.")

    def apply_unit(
        self, unit: str, state: InterpreterState, arm: str = "main",
    ) -> tuple[InterpreterState, StepGroundResult]:
        if (arm not in self.arm_names
                or (unit not in TRANSLATION_UNITS and unit not in ROTATION_UNITS)):
            return super().apply_unit(unit, state, arm)

        before = self.fk_tcp(state, arm)
        setpoint = state.tcp_setpoint[arm]
        clamped = False
        if unit in TRANSLATION_UNITS:
            raw = setpoint.p + state.step_m * np.array(
                TRANSLATION_UNITS[unit], dtype=float,
            )
            position = np.clip(raw, *self.workspace_bounds)
            clamped = not np.allclose(raw, position)
            target = Transform(R=setpoint.R.copy(), p=position)
        else:
            rotation = axis_angle_to_matrix(
                np.array(ROTATION_UNITS[unit], dtype=float),
                state.rot_step_rad,
            )
            target = Transform(
                R=rotation @ setpoint.R, p=setpoint.p.copy(),
            )

        result = self.solvers[arm].solve_ik(
            target_world=target,
            q_seed=state.arm_q[arm].copy(),
            T_root=self._root(state, arm),
            pos_tol_m=1e-4,
            rot_tol_rad=1e-3,
            rot_weight=1.0,
            max_iters=120,
            num_restarts=8,
        )
        # This arm has seven DoF, so preserve the requested orientation.
        accepted = bool(
            result.success and np.all(np.isfinite(result.q))
            and np.isfinite(result.pos_err_m)
            and np.isfinite(result.rot_err_rad)
        )
        if accepted:
            state.arm_q[arm] = result.q.copy()
            state.tcp_setpoint[arm] = self.fk_tcp(state, arm)
        return state, StepGroundResult(
            unit=unit,
            arm=arm,
            accepted=accepted,
            clamped_by_workspace=clamped,
            ik_pos_err_m=float(result.pos_err_m),
            delta_tcp_world=self.fk_tcp(state, arm).p - before.p,
            note="" if accepted else "IK target not reached.",
        )


def build_interpreter(urdf_path: str) -> EmbodimentInterpreter:
    return Kinova3Interpreter(urdf_path)
