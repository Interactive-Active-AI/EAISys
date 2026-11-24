import numpy as np

from eaisys.harness.action_space import ROTATION_UNITS, TRANSLATION_UNITS
from eaisys.harness.interpreter import EmbodimentInterpreter, InterpreterState, StepGroundResult
from eaisys.types import Transform, axis_angle_to_matrix
from harnesses.common.urdf_interpreter_base import ChainConfig, GenericURDFInterpreter


class SawyerInterpreter(GenericURDFInterpreter):
    def apply_unit(
        self, unit: str, state: InterpreterState, arm: str = "main",
    ) -> tuple[InterpreterState, StepGroundResult]:
        translation = unit in TRANSLATION_UNITS
        if arm not in self.arm_names or (not translation and unit not in ROTATION_UNITS):
            return super().apply_unit(unit, state, arm)

        setpoint = state.tcp_setpoint[arm]
        clamped = False
        if translation:
            raw = setpoint.p + state.step_m * np.array(TRANSLATION_UNITS[unit])
            position = np.clip(raw, *self.workspace_bounds)
            clamped = not np.allclose(raw, position)
            target = Transform(R=setpoint.R.copy(), p=position)
        else:
            axis = np.array(ROTATION_UNITS[unit], dtype=float)
            rotation = axis_angle_to_matrix(axis, state.rot_step_rad) @ setpoint.R
            target = Transform(R=rotation, p=setpoint.p.copy())

        result = self.solvers[arm].solve_ik(
            target_world=target, q_seed=state.arm_q[arm].copy(),
            T_root=self._root(state, arm), pos_tol_m=1.5e-3, rot_tol_rad=2.5e-2,
            rot_weight=1.0, max_iters=120, num_restarts=8,
        )
        accepted = bool(
            result.success and np.all(np.isfinite(result.q)) and np.isfinite(result.pos_err_m)
        )
        delta = np.zeros(3)
        if accepted:
            before = self.fk_tcp(state, arm).p
            state.arm_q[arm] = result.q.copy()
            state.tcp_setpoint[arm] = self.fk_tcp(state, arm)
            delta = state.tcp_setpoint[arm].p - before

        return state, StepGroundResult(
            unit=unit, arm=arm, accepted=accepted, clamped_by_workspace=clamped,
            ik_pos_err_m=float(result.pos_err_m), delta_tcp_world=delta,
            note="" if accepted else "IK target not reached.",
        )


def build_interpreter(urdf_path: str) -> EmbodimentInterpreter:
    chain = ChainConfig(
        name="main", root_link="right_arm_base_link", tcp_link="sawyer_mujoco_tcp",
        joint_names=tuple(f"right_j{i}" for i in range(7)),
        # A 90-degree elbow bend puts the downward TCP near (0.63, 0.16, 0.22) m.
        home_q=[0.0, -np.pi / 3, 0.0, np.pi / 2, 0.0, np.pi / 3, 0.0],
    )
    return SawyerInterpreter(
        urdf_path=urdf_path, robot_id="sawyer_mujoco", chains=[chain],
        supports_mobile_base=False, torso_joints=(),
        max_gripper_width_m=2.0 * 0.032333,
        workspace_bounds=(np.array([0.40, -0.08, 0.08]), np.array([0.82, 0.40, 0.42])),
    )
