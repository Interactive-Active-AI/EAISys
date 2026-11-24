"""Zero-shot iiwa grounding with an IK-reachable forward workspace boundary."""

import numpy as np

from eaisys.urdf import forward_kinematics_tree
from eaisys.harness.action_space import STEP_MODE_UNITS
from eaisys.harness.interpreter import EmbodimentInterpreter
from harnesses.common.urdf_interpreter_base import ChainConfig, GenericURDFInterpreter


def build_interpreter(urdf_path: str) -> EmbodimentInterpreter:
    names = tuple(f"iiwa_joint_{i}" for i in range(1, 8))

    def make(home=None):
        return GenericURDFInterpreter(
            urdf_path=urdf_path,
            robot_id="kuka_iiwa",
            chains=[ChainConfig(
                name="main",
                root_link="iiwa_link_0",
                tcp_link="iiwa_tcp",
                joint_names=names,
                home_q=home,
            )],
            supports_mobile_base=False,
            torso_joints=(),
            max_gripper_width_m=2.0 * 0.042,
            workspace_bounds=None,
        )

    probe = make()
    solver = probe.solvers["main"]
    root = forward_kinematics_tree(probe.tree, {})["iiwa_link_0"]
    nodes = [probe.tree.joints[name] for name in names]
    lo = np.maximum([joint.lower for joint in nodes], -2.0) + 1e-6
    hi = np.minimum([joint.upper for joint in nodes], 2.0) - 1e-6
    samples = np.random.default_rng(7).uniform(lo, hi, size=(128, 7))
    candidates = [probe.reset().arm_q["main"], *samples]

    ranked = []
    for q in candidates:
        p = solver.fk(q, T_root=root).p
        if not np.all(np.isfinite(p)) or p[0] <= 0.28 or p[2] <= 0.14:
            continue
        if abs(q[3]) < 0.4:  # Require a bent elbow.
            continue
        score = solver.manipulability(q, T_root=root)
        if np.isfinite(score) and score > 1e-6:
            ranked.append((float(score), q))

    for _, q in sorted(ranked, key=lambda item: item[0], reverse=True):
        robot = make(q.tolist())
        home_p = robot.reset().tcp_setpoint["main"].p
        robot.workspace_bounds = (home_p - 0.12, home_p + 0.12)

        # An unreachable face lets IK stall before any incremental step clips.
        # Require accepted forward motion through the boundary in every mode.
        reachable = True
        for mode in STEP_MODE_UNITS:
            state, _ = robot.apply_unit(mode, robot.reset())
            for _ in range(int(np.ceil(1.2 / state.step_m))):
                state, result = robot.apply_unit("MV_FWD", state)
                if not result.accepted:
                    reachable = False
                    break
                if result.clamped_by_workspace:
                    break
            else:
                reachable = False
            if not reachable:
                break

        if reachable:
            return robot

    raise ValueError(
        "No well-conditioned bent-elbow home with a reachable forward "
        "workspace boundary was found."
    )
