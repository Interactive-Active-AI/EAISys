"""Zero-shot Kinova Gen3 grounding with a reachable forward workspace face."""

import numpy as np

from eaisys.types import Transform, axis_angle_to_matrix
from eaisys.urdf import KinematicChainSolver
from eaisys.harness.action_space import TRANSLATION_UNITS, ROTATION_UNITS
from eaisys.harness.interpreter import (
    EmbodimentInterpreter, InterpreterState, StepGroundResult,
)
from harnesses.common.urdf_interpreter_base import (
    ChainConfig, GenericURDFInterpreter,
)

ARM_JOINTS = tuple(f"joint_{i}" for i in range(1, 8))


class KinovaGen3Interpreter(GenericURDFInterpreter):
    def __init__(self, urdf_path: str):
        super().__init__(
            urdf_path=urdf_path,
            robot_id="kinova_gen3",
            chains=[ChainConfig(
                name="main", root_link="base_link", tcp_link="gen3_tcp",
                joint_names=ARM_JOINTS, home_q=None,
            )],
            supports_mobile_base=False,
            torso_joints=(),
            max_gripper_width_m=0.085,
            workspace_bounds=None,
        )
        nodes = [self.tree.joints[n] for n in ARM_JOINTS]
        lower = np.array([j.lower for j in nodes], dtype=float)
        upper = np.array([j.upper for j in nodes], dtype=float)
        continuous = np.array([j.is_continuous for j in nodes], dtype=bool)
        a = np.where(continuous, -2.0, np.maximum(lower, -2.0))
        b = np.where(continuous, 2.0, np.minimum(upper, 2.0))
        rng = np.random.default_rng(7)
        candidates = [
            self._homes["main"].copy(),
            *rng.uniform(a + 1e-5, b - 1e-5, size=(96, 7)),
        ]
        T_root = self._root(self.reset(), "main")

        # A distant box face may be unreachable while holding orientation:
        # IK then stops progress before clipping ever occurs. Certify the face.
        for home in candidates:
            if abs(home[3]) < 0.4:
                continue
            solver = KinematicChainSolver(
                tree=self.tree, root_link="base_link", tcp_link="gen3_tcp",
                actuated_joint_names=ARM_JOINTS,
                lower_limits=lower, upper_limits=upper,
                continuous_mask=continuous, home_q=home.copy(),
            )
            quality = solver.manipulability(home, T_root=T_root)
            p = solver.fk(home, T_root=T_root).p
            if (not np.isfinite(quality) or quality <= 1e-6
                    or not np.all(np.isfinite(p))
                    or p[0] <= 0.28 or p[2] <= 0.14):
                continue
            self.solvers["main"] = solver
            self._homes["main"] = home.copy()
            self.workspace_bounds = (
                p - np.array([0.16, 0.20, 0.16]),
                p + np.array([0.11, 0.20, 0.20]),
            )
            # Check every step mode and a single runaway command using the
            # actual action implementation, including achieved-pose feedback.
            viable = True
            for step in (0.01, 0.02, 0.04, 1.2):
                probe = self.reset()
                probe.step_m = step
                reached_face = False
                for _ in range(int(np.ceil(0.13 / step)) + 2):
                    probe, result = self.apply_unit("MV_FWD", probe)
                    if not result.accepted:
                        break
                    if result.clamped_by_workspace:
                        reached_face = True
                        break
                if not reached_face:
                    viable = False
                    break
            if viable:
                return
        raise ValueError("No nonsingular home with a reachable forward face found.")

    def apply_unit(
        self, unit: str, state: InterpreterState, arm: str = "main",
    ) -> tuple[InterpreterState, StepGroundResult]:
        if (arm not in self.arm_names
                or unit not in TRANSLATION_UNITS and unit not in ROTATION_UNITS):
            return super().apply_unit(unit, state, arm)

        before = self.fk_tcp(state, arm)
        setpoint = state.tcp_setpoint[arm]
        clamped = False
        if unit in TRANSLATION_UNITS:
            raw = setpoint.p + state.step_m * np.array(
                TRANSLATION_UNITS[unit], dtype=float,
            )
            p = np.clip(raw, *self.workspace_bounds)
            clamped = not np.allclose(raw, p)
            target = Transform(R=setpoint.R.copy(), p=p)
        else:
            R = axis_angle_to_matrix(
                np.array(ROTATION_UNITS[unit], dtype=float),
                state.rot_step_rad,
            ) @ setpoint.R
            target = Transform(R=R, p=setpoint.p.copy())

        solver = self.solvers[arm]
        result = solver.solve_ik(
            target_world=target,
            q_seed=state.arm_q[arm].copy(),
            T_root=self._root(state, arm),
            pos_tol_m=1.5e-3, rot_tol_rad=2.5e-2,
            rot_weight=1.0, max_iters=120, num_restarts=8,
        )
        valid = (
            result.success
            and np.all(np.isfinite(result.q))
            and np.isfinite(result.pos_err_m)
            and np.isfinite(result.rot_err_rad)
        )
        if not valid:
            return state, StepGroundResult(
                unit=unit, arm=arm, accepted=False,
                clamped_by_workspace=clamped,
                ik_pos_err_m=float(result.pos_err_m),
                note="IK target not reached.",
            )
        state.arm_q[arm] = result.q.copy()
        achieved = self.fk_tcp(state, arm)
        state.tcp_setpoint[arm] = achieved
        return state, StepGroundResult(
            unit=unit, arm=arm, accepted=True,
            clamped_by_workspace=clamped,
            ik_pos_err_m=float(result.pos_err_m),
            delta_tcp_world=achieved.p - before.p,
        )


def build_interpreter(urdf_path: str) -> EmbodimentInterpreter:
    return KinovaGen3Interpreter(urdf_path)
