"""ARX X5A: world-frame arm increments and heading-frame mobile-base increments."""

import numpy as np

from eaisys.types import Transform, axis_angle_to_matrix
from eaisys.harness.action_space import ROTATION_UNITS, TRANSLATION_UNITS
from eaisys.harness.interpreter import (
    EmbodimentInterpreter, InterpreterState, StepGroundResult,
)
from harnesses.common.urdf_interpreter_base import (
    ChainConfig, GenericURDFInterpreter,
)


class ARXMobileInterpreter(GenericURDFInterpreter):
    robot_id = "arx_mobile"
    arm_names = ("left", "right")
    supports_mobile_base = True
    max_gripper_width_m = 0.088

    def __init__(self, urdf_path: str):
        chains = [
            ChainConfig(
                name=arm,
                root_link=f"{arm}_base_link",
                tcp_link=f"{arm}_tcp",
                joint_names=tuple(f"{arm}_joint{i}" for i in range(1, 7)),
                home_q=[0.0, 0.8, 0.8, 0.0, 0.0, 0.0],
            )
            for arm in self.arm_names
        ]
        # The common constructor verifies interior limits and manipulability.
        # Its automatic workspace provides >=0.18 m reset margins on every axis.
        super().__init__(
            urdf_path=urdf_path,
            robot_id="arx_mobile",
            chains=chains,
            supports_mobile_base=True,
            torso_joints=("torso_lift_joint",),
            max_gripper_width_m=0.088,
            workspace_bounds=None,
        )

    def arm_joint_names(self, arm: str = "left") -> tuple[str, ...]:
        return super().arm_joint_names(arm)

    def tcp_link_name(self, arm: str = "left") -> str:
        return super().tcp_link_name(arm)

    def reset(self) -> InterpreterState:
        return super().reset()

    def fk_tcp(self, state: InterpreterState, arm: str = "left") -> Transform:
        # Inherited FK includes torso height and the selected arm's mount.
        return super().fk_tcp(state, arm)

    def apply_unit(
        self, unit: str, state: InterpreterState, arm: str = "left",
    ) -> tuple[InterpreterState, StepGroundResult]:
        translation = unit in TRANSLATION_UNITS
        rotation = unit in ROTATION_UNITS
        if arm not in self.arm_names or not (translation or rotation):
            # Handles step modes, normalized gripper commands, HOLD/DONE,
            # and bounded heading-frame base motion with both TCPs refreshed.
            return super().apply_unit(unit, state, arm)

        before = self.fk_tcp(state, arm)
        setpoint = state.tcp_setpoint[arm]
        clamped = False
        if translation:
            raw = setpoint.p + state.step_m * np.array(
                TRANSLATION_UNITS[unit], dtype=float,
            )
            p = raw.copy()
            if not self.supports_mobile_base:
                p = np.clip(raw, *self.workspace_bounds)
                clamped = not np.allclose(raw, p)
            target = Transform(R=setpoint.R.copy(), p=p)
        else:
            increment = axis_angle_to_matrix(
                np.array(ROTATION_UNITS[unit], dtype=float),
                state.rot_step_rad,
            )
            target = Transform(R=increment @ setpoint.R, p=setpoint.p.copy())

        solver = self.solvers[arm]
        options = dict(
            target_world=target,
            q_seed=state.arm_q[arm].copy(),
            # Includes the torso and mount; virtual base motion is applied once.
            T_root=self._root(state, arm),
            pos_tol_m=1.5e-3,
            rot_tol_rad=2.5e-2,
            max_iters=120,
            num_restarts=8,
        )
        result = solver.solve_ik(**options, rot_weight=1.0)
        relaxed = (
            translation
            and len(self.arm_joint_names(arm)) == 6
            and not result.success
        )
        if relaxed:
            result = solver.solve_ik(**options, rot_weight=0.15)

        accepted = result.pos_err_m <= 0.003 if relaxed else result.success
        accepted = bool(
            accepted
            and np.all(np.isfinite(result.q))
            and np.isfinite(result.pos_err_m)
            and np.isfinite(result.rot_err_rad)
        )
        note = "IK target not reached."
        if accepted:
            state.arm_q[arm] = result.q.copy()
            state.tcp_setpoint[arm] = self.fk_tcp(state, arm)
            note = "Orientation relaxed." if relaxed else ""

        return state, StepGroundResult(
            unit=unit,
            arm=arm,
            accepted=accepted,
            clamped_by_workspace=clamped,
            ik_pos_err_m=float(result.pos_err_m),
            delta_tcp_world=self.fk_tcp(state, arm).p - before.p,
            note=note,
        )


def build_interpreter(urdf_path: str) -> EmbodimentInterpreter:
    return ARXMobileInterpreter(urdf_path)
