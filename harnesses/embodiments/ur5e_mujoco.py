import numpy as np

from eaisys.harness.interpreter import EmbodimentInterpreter
from harnesses.common.urdf_interpreter_base import (
    ChainConfig,
    GenericURDFInterpreter,
)


def build_interpreter(urdf_path: str) -> EmbodimentInterpreter:
    joint_names = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    # These URDF joint frames give a bent elbow, nonsingular wrist,
    # and room to extend forward without straightening the arm.
    nominal = np.array([
        0.0, -np.pi / 2, np.pi / 2,
        -np.pi / 2, np.pi / 2, 0.0,
    ])
    rng = np.random.default_rng(0)
    candidates = rng.uniform(nominal - 0.08, nominal + 0.08, size=(16, 6))

    for home in candidates:
        # The common implementation validates limits and manipulability,
        # implements all five methods, and supplies world-frame root FK/IK.
        robot = GenericURDFInterpreter(
            urdf_path=urdf_path,
            robot_id="ur5e_mujoco",
            chains=[
                ChainConfig(
                    name="main",
                    root_link="fixed_base_link",
                    tcp_link="ur5e_mujoco_tcp",
                    joint_names=joint_names,
                    home_q=home.tolist(),
                ),
            ],
            supports_mobile_base=False,
            torso_joints=(),
            # Each opposing finger travels 0.0425 m; mimic adds no arm DOF.
            max_gripper_width_m=0.085,
            workspace_bounds=None,
        )
        probe = robot.reset()
        p_home = probe.tcp_setpoint["main"].p.copy()
        margin = np.full(3, 0.12)
        robot.workspace_bounds = (p_home - margin, p_home + margin)

        # Require an actually reachable forward boundary: a larger box can
        # let IK stall before repeated forward actions ever report clipping.
        # Probe state is discarded; reset still returns the sampled home.
        probe.step_m = 0.04
        for _ in range(8):
            probe, result = robot.apply_unit("MV_FWD", probe, "main")
            if not result.accepted:
                break
            if result.clamped_by_workspace:
                return robot

    raise ValueError(
        "No nonsingular home with a reachable forward workspace boundary."
    )
