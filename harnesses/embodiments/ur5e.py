"""UR5e grounding with a sampled bent home and a reachable workspace boundary."""

import numpy as np

from eaisys.types import Transform
from eaisys.urdf import (
    KinematicChainSolver,
    forward_kinematics_tree,
    parse_robot_file,
)
from eaisys.harness.interpreter import EmbodimentInterpreter
from harnesses.common.urdf_interpreter_base import (
    ChainConfig,
    GenericURDFInterpreter,
)


ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


class UR5eInterpreter(GenericURDFInterpreter):
    def __init__(self, urdf_path: str):
        tree = parse_robot_file(urdf_path)
        nodes = [tree.joints[name] for name in ARM_JOINTS]
        lower = np.array([j.lower for j in nodes], dtype=float)
        upper = np.array([j.upper for j in nodes], dtype=float)
        continuous = np.array([j.is_continuous for j in nodes], dtype=bool)

        # The negative-X arm links face forward after a pi shoulder-pan turn.
        # Bent elbow and wrist angles avoid the straight-arm singularities.
        nominal = np.array([
            np.pi, -np.pi / 2, np.pi / 2,
            -np.pi / 2, -np.pi / 2, 0.0,
        ])
        root = forward_kinematics_tree(tree, {})["base_link"]
        probe = KinematicChainSolver(
            tree=tree,
            root_link="base_link",
            tcp_link="ur5e_tcp",
            actuated_joint_names=ARM_JOINTS,
            lower_limits=lower,
            upper_limits=upper,
            continuous_mask=continuous,
            home_q=nominal.copy(),
        )

        rng = np.random.default_rng(0)
        candidates = nominal + rng.uniform(-0.08, 0.08, size=(32, 6))
        ranked = []
        for q in candidates:
            if np.any(((q <= lower) | (q >= upper)) & ~continuous):
                continue
            quality = probe.manipulability(q, T_root=root)
            if np.isfinite(quality) and quality > 1e-6:
                ranked.append((float(quality), q.copy()))

        margin = np.array([0.12, 0.15, 0.12])
        home = center = None
        for _, q in sorted(ranked, key=lambda item: item[0], reverse=True):
            pose = probe.fk(q, T_root=root)
            target = Transform(
                R=pose.R.copy(),
                p=pose.p + np.array([margin[0], 0.0, 0.0]),
            )
            result = probe.solve_ik(
                target_world=target,
                q_seed=q.copy(),
                T_root=root,
                pos_tol_m=1.5e-3,
                rot_tol_rad=2.5e-2,
                rot_weight=1.0,
                max_iters=120,
                num_restarts=8,
            )
            if result.success and np.all(np.isfinite(result.q)):
                home, center = q, pose.p.copy()
                break
        if home is None:
            raise ValueError("No conditioned home with a reachable forward boundary.")

        # Keep the forward face reachable: otherwise repeated rejected IK steps
        # could stall before the workspace boundary and never report clamping.
        super().__init__(
            urdf_path=urdf_path,
            robot_id="ur5e",
            chains=[ChainConfig(
                name="main",
                root_link="base_link",
                tcp_link="ur5e_tcp",
                joint_names=ARM_JOINTS,
                home_q=home.tolist(),
            )],
            supports_mobile_base=False,
            torso_joints=(),
            max_gripper_width_m=float(
                tree.joints["left_finger_joint"].upper
                + tree.joints["right_finger_joint"].upper
            ),
            workspace_bounds=(center - margin, center + margin),
        )


def build_interpreter(urdf_path: str) -> EmbodimentInterpreter:
    return UR5eInterpreter(urdf_path)
