import numpy as np

from eaisys.harness.action_space import ROTATION_UNITS, TRANSLATION_UNITS
from eaisys.harness.interpreter import EmbodimentInterpreter, InterpreterState, StepGroundResult
from eaisys.types import Transform, axis_angle_to_matrix
from harnesses.common.urdf_interpreter_base import ChainConfig, GenericURDFInterpreter


class UR5eWipeInterpreter(GenericURDFInterpreter):
    def apply_unit(
        self, unit: str, state: InterpreterState, arm: str = "main",
    ) -> tuple[InterpreterState, StepGroundResult]:
        translation = unit in TRANSLATION_UNITS
        if arm not in self.arm_names or not (translation or unit in ROTATION_UNITS):
            return super().apply_unit(unit, state, arm)

        setpoint = state.tcp_setpoint[arm]
        p, R = setpoint.p.copy(), setpoint.R.copy()
        clamped = False
        if translation:
            raw = p + state.step_m * np.asarray(TRANSLATION_UNITS[unit], dtype=float)
            p = np.clip(raw, *self.workspace_bounds)
            clamped = not np.allclose(raw, p)
        else:
            axis = np.asarray(ROTATION_UNITS[unit], dtype=float)
            R = axis_angle_to_matrix(axis, state.rot_step_rad) @ R

        solve_ik = self.solvers[arm].solve_ik
        options = dict(
            target_world=Transform(R=R, p=p), q_seed=state.arm_q[arm].copy(),
            T_root=self._root(state, arm), pos_tol_m=1.5e-3, rot_tol_rad=2.5e-2,
            max_iters=120, num_restarts=8,
        )
        result = solve_ik(**options, rot_weight=1.0)
        relaxed = translation and not result.success
        if relaxed:
            result = solve_ik(**options, rot_weight=0.15)

        error = float(result.pos_err_m)
        reached = error <= 0.003 if relaxed else result.success
        accepted = bool(reached and np.isfinite(error) and np.isfinite(result.q).all())
        delta = np.zeros(3)
        note = "IK target not reached."
        if accepted:
            before = self.fk_tcp(state, arm).p
            state.arm_q[arm] = result.q.copy()
            after = self.fk_tcp(state, arm)
            state.tcp_setpoint[arm] = after
            delta = after.p - before
            note = "Orientation relaxed." if relaxed else ""

        return state, StepGroundResult(
            unit=unit, arm=arm, accepted=accepted, clamped_by_workspace=clamped,
            ik_pos_err_m=error, delta_tcp_world=delta, note=note,
        )


def build_interpreter(urdf_path: str) -> EmbodimentInterpreter:
    pi = np.pi
    chain = ChainConfig(
        name="main", root_link="fixed_base_link", tcp_link="ur5e_wipe_mujoco_tcp",
        joint_names=(
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ),
        home_q=[0.0, -pi / 2, 2 * pi / 3, -2 * pi / 3, -pi / 2, 0.0],
    )
    # Home places the downward-facing TCP near (0.439, 0.134, 0.264) m.
    # The forward face at x = 0.57 m is reachable with that orientation held.
    return UR5eWipeInterpreter(
        urdf_path=urdf_path, robot_id="ur5e_wipe_mujoco", chains=[chain],
        supports_mobile_base=False, torso_joints=(),
        max_gripper_width_m=0.0,  # The fixed wiping tool has zero jaw travel.
        workspace_bounds=(np.array([0.30, 0.00, 0.13]), np.array([0.57, 0.27, 0.40])),
    )
