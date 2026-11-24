"""World-frame semantic actions for the fixed-base ARX bimanual station."""

import numpy as np

from eaisys.harness.interpreter import EmbodimentInterpreter
from harnesses.common.urdf_interpreter_base import (
    ChainConfig,
    GenericURDFInterpreter,
)


def build_interpreter(urdf_path: str) -> EmbodimentInterpreter:
    # The shared implementation checks limits and home manipulability,
    # includes each mount transform in FK/IK, and implements all actions.
    chains = [
        ChainConfig(
            name="left",
            root_link="left_base_link",
            tcp_link="left_tcp",
            joint_names=(
                "left_joint1", "left_joint2", "left_joint3",
                "left_joint4", "left_joint5", "left_joint6",
            ),
            home_q=[0.0, 0.8, 0.8, 0.0, 0.0, 0.0],
        ),
        ChainConfig(
            name="right",
            root_link="right_base_link",
            tcp_link="right_tcp",
            joint_names=(
                "right_joint1", "right_joint2", "right_joint3",
                "right_joint4", "right_joint5", "right_joint6",
            ),
            home_q=[0.0, 0.8, 0.8, 0.0, 0.0, 0.0],
        ),
    ]
    interpreter = GenericURDFInterpreter(
        urdf_path=urdf_path,
        robot_id="arx_bimanual",
        chains=chains,
        supports_mobile_base=False,
        torso_joints=(),
        max_gripper_width_m=0.088,
    )

    # Derive world bounds from actual FK, including the ±0.25 m mounts.
    home = interpreter.reset()
    points = np.stack([
        interpreter.fk_tcp(home, arm).p for arm in interpreter.arm_names
    ])
    # A reachable forward boundary prevents repeated IK rejection from
    # stalling motion before workspace clipping can occur.
    # Both homes have at least 0.12 m clearance on every box face.
    padding = np.array([0.12, 0.15, 0.15], dtype=float)
    interpreter.workspace_bounds = (
        points.min(axis=0) - padding,
        points.max(axis=0) + padding,
    )
    return interpreter
