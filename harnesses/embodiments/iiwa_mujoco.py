import numpy as np

from eaisys.harness.action_space import ROTATION_UNITS, TRANSLATION_UNITS
from eaisys.harness.interpreter import EmbodimentInterpreter, InterpreterState, StepGroundResult
from eaisys.types import Transform, axis_angle_to_matrix
from harnesses.common.urdf_interpreter_base import ChainConfig, GenericURDFInterpreter


class IiwaInterpreter(GenericURDFInterpreter):
    def apply_unit(
        self, unit: str, state: InterpreterState, arm: str = "main",
    ) -> tuple[InterpreterState, StepGroundResult]:
        translation = unit in TRANSLATION_UNITS
        if arm not in self.arm_names or (not translation and unit not in ROTATION_UNITS):
            return super().apply_unit(unit, state, arm)

        setpoint = state.tcp_setpoint[arm]
        R, p = setpoint.R.copy(), setpoint.p.copy()
        clamped = False
        if translation:
            p_raw = p + state.step_m * np.array(TRANSLATION_UNITS[unit])
            p = np.clip(p_raw, *self.workspace_bounds)
            clamped = not np.allclose(p_raw, p)
        else:
            axis = np.array(ROTATION_UNITS[unit], dtype=float)
            R = axis_angle_to_matrix(axis, state.rot_step_rad) @ R

        result = self.solvers[arm].solve_ik(
            target_world=Transform(R=R, p=p), q_seed=state.arm_q[arm].copy(),
            T_root=self._root(state, arm), pos_tol_m=1.5e-3, rot_tol_rad=2.5e-2,
            rot_weight=1.0, max_iters=120, num_restarts=8,
        )
        accepted = bool(
            result.success
            and np.isfinite(result.pos_err_m)
            and np.all(np.isfinite(result.q))
        )
        delta = np.zeros(3)
        if accepted:
            before = self.fk_tcp(state, arm)
            state.arm_q[arm] = result.q.copy()
            state.tcp_setpoint[arm] = self.fk_tcp(state, arm)
            delta = state.tcp_setpoint[arm].p - before.p

        return state, StepGroundResult(
            unit=unit, arm=arm, accepted=accepted,
            clamped_by_workspace=clamped, ik_pos_err_m=float(result.pos_err_m),
            delta_tcp_world=delta, note="" if accepted else "IK target not reached.",
        )


def build_interpreter(urdf_path: str) -> EmbodimentInterpreter:
    chain = ChainConfig(
        name="main", root_link="fixed_base_link", tcp_link="iiwa_mujoco_tcp",
        joint_names=tuple(f"joint_{i}" for i in range(1, 8)),
        # Bent elbow and wrist. The base checks limits and manipulability.
        home_q=[0.0, 0.4, 0.0, -1.8, 0.0, -1.2, 0.0],
    )
    return IiwaInterpreter(
        urdf_path=urdf_path, robot_id="iiwa_mujoco", chains=[chain],
        supports_mobile_base=False, torso_joints=(),
        max_gripper_width_m=2.0 * 0.0425,
    )
