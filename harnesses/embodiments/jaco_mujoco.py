import numpy as np

from eaisys.harness.action_space import ROTATION_UNITS, TRANSLATION_UNITS
from eaisys.harness.interpreter import (
    EmbodimentInterpreter, InterpreterState, StepGroundResult,
)
from eaisys.types import Transform, axis_angle_to_matrix
from harnesses.common.urdf_interpreter_base import ChainConfig, GenericURDFInterpreter


class JacoInterpreter(GenericURDFInterpreter):
    def apply_unit(
        self, unit: str, state: InterpreterState, arm: str = "main",
    ) -> tuple[InterpreterState, StepGroundResult]:
        translation = unit in TRANSLATION_UNITS
        if arm not in self.arm_names or (not translation and unit not in ROTATION_UNITS):
            return super().apply_unit(unit, state, arm)

        before = self.fk_tcp(state, arm)
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
            result.success and np.isfinite(result.pos_err_m)
            and np.all(np.isfinite(result.q))
        )
        after = before
        if accepted:
            state.arm_q[arm] = result.q.copy()
            after = self.fk_tcp(state, arm)
            state.tcp_setpoint[arm] = after

        return state, StepGroundResult(
            unit=unit, arm=arm, accepted=accepted, clamped_by_workspace=clamped,
            ik_pos_err_m=float(result.pos_err_m), delta_tcp_world=after.p - before.p,
            note="" if accepted else "IK target not reached.",
        )


def build_interpreter(urdf_path: str) -> EmbodimentInterpreter:
    chain = ChainConfig(
        name="main", root_link="j2s7s300_link_0", tcp_link="jaco_mujoco_tcp",
        joint_names=tuple(f"j2s7s300_joint_{i}" for i in range(1, 8)),
        home_q=[0.0, 2.8, 0.0, 4.7, 0.0, 2.15, 0.0],
    )
    interpreter = JacoInterpreter(
        urdf_path=urdf_path, robot_id="jaco_mujoco", chains=[chain],
        supports_mobile_base=False, torso_joints=(), max_gripper_width_m=2 * 0.0425,
    )
    home = interpreter.reset().tcp_setpoint["main"].p
    lower, upper = interpreter.workspace_bounds
    upper = upper.copy()
    # The bent elbow leaves enough extension to reach this wall at fixed TCP orientation.
    upper[0] = home[0] + 0.12
    interpreter.workspace_bounds = (lower, upper)
    return interpreter
