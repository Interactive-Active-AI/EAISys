import numpy as np

from eaisys.types import Transform, axis_angle_to_matrix
from eaisys.urdf import (
    KinematicChainSolver, forward_kinematics_tree, parse_robot_file,
)
from eaisys.harness.action_space import (
    MOBILE_BASE_UNITS, ROTATION_UNITS, STEP_MODE_UNITS, TRANSLATION_UNITS,
)
from eaisys.harness.interpreter import (
    EmbodimentInterpreter, InterpreterState, StepGroundResult,
)


class PandaInterpreter(EmbodimentInterpreter):
    robot_id = "franka_panda"
    arm_names = ("main",)
    supports_mobile_base = False
    max_gripper_width_m = 0.08

    def __init__(self, urdf_path: str):
        self.tree = parse_robot_file(urdf_path)
        self.root_link = "panda_link0"
        self.tcp_link = "panda_hand_tcp"
        self.joint_names = tuple(f"panda_joint{i}" for i in range(1, 8))
        nodes = [self.tree.joints[name] for name in self.joint_names]
        self.lower = np.array([node.lower for node in nodes], dtype=float)
        self.upper = np.array([node.upper for node in nodes], dtype=float)
        continuous = np.array([node.is_continuous for node in nodes], dtype=bool)
        self.home_q = np.array(
            [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]
        )
        if not np.all((self.home_q > self.lower) & (self.home_q < self.upper)):
            raise ValueError("Provided bent-elbow home violates strict joint limits.")
        self.solver = KinematicChainSolver(
            tree=self.tree,
            root_link=self.root_link,
            tcp_link=self.tcp_link,
            actuated_joint_names=self.joint_names,
            lower_limits=self.lower,
            upper_limits=self.upper,
            continuous_mask=continuous,
            home_q=self.home_q,
        )
        initial = self.reset()
        T_root = self._root_transform(initial)
        manipulability = self.solver.manipulability(self.home_q, T_root=T_root)
        if not np.isfinite(manipulability) or manipulability <= 1e-6:
            raise ValueError("Provided home failed the manipulability check.")
        # Conservative world-frame operating box with 20 cm home clearance.
        center = initial.tcp_setpoint["main"].p.copy()
        extent = np.array([0.20, 0.20, 0.20])
        self.workspace_bounds = (center - extent, center + extent)

    def _check_arm(self, arm: str):
        if arm not in self.arm_names:
            raise ValueError(f"Unknown arm: {arm}")

    def arm_joint_names(self, arm: str = "main") -> tuple[str, ...]:
        self._check_arm(arm)
        return self.joint_names

    def tcp_link_name(self, arm: str = "main") -> str:
        self._check_arm(arm)
        return self.tcp_link

    def _root_transform(self, state: InterpreterState) -> Transform:
        x, y, yaw = state.base_se2
        T_base = Transform.from_xyz_rpy(
            xyz=(float(x), float(y), 0.0), rpy=(0.0, 0.0, float(yaw))
        )
        torso_joint_dict = {}  # This Panda has no torso joints.
        return forward_kinematics_tree(
            self.tree, torso_joint_dict, base_transform=T_base
        )[self.root_link]

    def reset(self) -> InterpreterState:
        state = InterpreterState(
            arm_q={"main": self.home_q.copy()},
            tcp_setpoint={},
            gripper_open={"main": 1.0},
        )
        state.tcp_setpoint["main"] = self.fk_tcp(state, "main")
        return state

    def fk_tcp(self, state: InterpreterState, arm: str = "main") -> Transform:
        self._check_arm(arm)
        return self.solver.fk(state.arm_q[arm], T_root=self._root_transform(state))

    def apply_unit(
        self, unit: str, state: InterpreterState, arm: str = "main"
    ) -> tuple[InterpreterState, StepGroundResult]:
        def finish(accepted=True, **kwargs):
            return state, StepGroundResult(
                unit=unit, arm=arm, accepted=accepted, **kwargs
            )

        if arm not in self.arm_names:
            return finish(False, note="Unknown arm.")
        if unit in STEP_MODE_UNITS:
            state.step_m = float(STEP_MODE_UNITS[unit])
            return finish()
        if unit in ("GRASP", "RELEASE"):
            # Openness 1 means 0.04 m per finger: 0.08 m total aperture.
            state.gripper_open[arm] = float(unit == "RELEASE")
            return finish()
        if unit in ("HOLD", "DONE"):
            return finish()
        if unit in MOBILE_BASE_UNITS:
            return finish(False, note="Panda has a fixed base.")

        current = state.tcp_setpoint[arm]
        position, rotation = current.p.copy(), current.R.copy()
        clamped = False
        if unit in TRANSLATION_UNITS:
            p_raw = current.p + state.step_m * np.array(TRANSLATION_UNITS[unit])
            position = np.clip(p_raw, *self.workspace_bounds)
            clamped = not np.allclose(p_raw, position)
        elif unit in ROTATION_UNITS:
            rotation = axis_angle_to_matrix(
                np.array(ROTATION_UNITS[unit], dtype=float), state.rot_step_rad
            ) @ current.R
        else:
            return finish(False, note="Unknown action unit.")

        target = Transform(R=rotation, p=position)
        options = dict(
            target_world=target,
            q_seed=state.arm_q[arm].copy(),
            T_root=self._root_transform(state),
            pos_tol_m=1.5e-3,
            rot_tol_rad=2.5e-2,
            max_iters=120,
            num_restarts=8,
        )
        ik = self.solver.solve_ik(**options, rot_weight=1.0)
        accepted = bool(ik.success)
        if not accepted and len(self.joint_names) == 6:
            options["q_seed"] = state.arm_q[arm].copy()
            ik = self.solver.solve_ik(**options, rot_weight=0.15)
            accepted = ik.pos_err_m <= 0.003
        if not accepted or not np.all(np.isfinite(ik.q)):
            return finish(
                False, clamped_by_workspace=clamped,
                ik_pos_err_m=float(ik.pos_err_m), note="IK target not reached.",
            )
        before = self.fk_tcp(state, arm).p.copy()
        state.arm_q[arm] = ik.q.copy()
        state.tcp_setpoint[arm] = self.fk_tcp(state, arm)
        return finish(
            clamped_by_workspace=clamped, ik_pos_err_m=float(ik.pos_err_m),
            delta_tcp_world=state.tcp_setpoint[arm].p - before,
        )


def build_interpreter(urdf_path: str) -> EmbodimentInterpreter:
    return PandaInterpreter(urdf_path)
