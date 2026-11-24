"""World-frame grounding for the fixed-base, dual-arm Franka."""

import numpy as np

from eaisys.types import Transform
from eaisys.harness.interpreter import (
    EmbodimentInterpreter, InterpreterState, StepGroundResult,
)
from harnesses.common.urdf_interpreter_base import (
    ChainConfig, GenericURDFInterpreter,
)


class DualFrankaInterpreter(GenericURDFInterpreter):
    robot_id = "dual_franka"
    arm_names = ("left", "right")
    supports_mobile_base = False
    max_gripper_width_m = 0.08

    def __init__(self, urdf_path: str):
        homes = {
            "left": [0.08, 0.03, 0.05, -1.75, 0.03, 1.57, -0.55],
            "right": [0.57, 0.18, -0.61, -1.58, -0.04, 1.48, 0.80],
        }
        chains = [
            ChainConfig(
                name=arm,
                root_link=f"{arm}_link0",
                tcp_link=f"{arm}_hand_tcp",
                joint_names=tuple(f"{arm}_joint{i}" for i in range(1, 8)),
                home_q=homes[arm],
            )
            for arm in self.arm_names
        ]
        # The backend validates home limits and manipulability and includes
        # each arm's fixed mount transform in world-frame FK and IK.
        super().__init__(
            urdf_path=urdf_path, robot_id=self.robot_id, chains=chains,
            supports_mobile_base=False, torso_joints=(),
            max_gripper_width_m=self.max_gripper_width_m,
            workspace_bounds=None,
        )
        state = self.reset()
        points = np.array([state.tcp_setpoint[a].p for a in self.arm_names])
        # Keep the forward boundary reachable with the home TCP orientations;
        # the generic +0.28 m extension lets IK stall before workspace clipping.
        padding = np.array([0.105, 0.12, 0.12])
        self.workspace_bounds = (
            points.min(axis=0) - padding, points.max(axis=0) + padding,
        )

    def arm_joint_names(self, arm: str = "left") -> tuple[str, ...]:
        return super().arm_joint_names(arm)

    def tcp_link_name(self, arm: str = "left") -> str:
        return super().tcp_link_name(arm)

    def reset(self) -> InterpreterState:
        return super().reset()

    def fk_tcp(self, state: InterpreterState, arm: str = "left") -> Transform:
        return super().fk_tcp(state, arm)

    def apply_unit(
        self, unit: str, state: InterpreterState, arm: str = "left",
    ) -> tuple[InterpreterState, StepGroundResult]:
        return super().apply_unit(unit, state, arm)


def build_interpreter(urdf_path: str) -> EmbodimentInterpreter:
    return DualFrankaInterpreter(urdf_path)
