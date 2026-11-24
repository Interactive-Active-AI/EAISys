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


class PandaOmronInterpreter(EmbodimentInterpreter):
    robot_id = "panda_omron"
    arm_names = ("main",)
    supports_mobile_base = True
    max_gripper_width_m = 0.08
    _names = tuple(f"panda_joint{i}" for i in range(1, 8))

    def __init__(self, urdf_path: str):
        self.tree = parse_robot_file(urdf_path)
        self._home = np.array([0., -0.7854, 0., -2.3562, 0., 1.5708, 0.7854])
        lower = np.array([self.tree.joints[n].lower for n in self._names])
        upper = np.array([self.tree.joints[n].upper for n in self._names])
        continuous = np.array(
            [self.tree.joints[n].is_continuous for n in self._names], dtype=bool,
        )
        if not np.all((lower < self._home) & (self._home < upper)):
            raise ValueError("Sidecar home posture must be strictly inside joint limits.")
        self.solver = KinematicChainSolver(
            tree=self.tree, root_link="panda_link0", tcp_link="panda_hand_tcp",
            actuated_joint_names=self._names, lower_limits=lower,
            upper_limits=upper, continuous_mask=continuous, home_q=self._home.copy(),
        )
        # Conservative TCP envelope from link lengths, base travel, and lift travel.
        reach = 0.333 + 0.316 + 0.0825 + np.hypot(0.0825, 0.384) + 0.088 + 0.2104
        spread = 0.15 + reach
        self.workspace_bounds = (
            np.array([-10. - spread, -10. - spread, 0.70 - 0.15 - reach]),
            np.array([10. + spread, 10. + spread, 0.70 + 0.35 + reach]),
        )
        initial = self.reset()
        quality = self.solver.manipulability(self._home, T_root=self._root(initial))
        if not np.isfinite(quality) or quality <= 1e-4:
            raise ValueError("Sidecar home posture failed the manipulability check.")
        lo, hi = self.workspace_bounds
        p = initial.tcp_setpoint["main"].p
        if np.any(p - lo < 0.10) or np.any(hi - p < 0.10):
            raise ValueError("Workspace must provide 0.10 m of home TCP margin.")

    def arm_joint_names(self, arm: str = "main") -> tuple[str, ...]:
        if arm != "main":
            raise ValueError(f"Unknown arm: {arm}")
        return self._names

    def tcp_link_name(self, arm: str = "main") -> str:
        self.arm_joint_names(arm)
        return "panda_hand_tcp"

    def _root(self, state: InterpreterState) -> Transform:
        x, y, yaw = state.base_se2
        T_base = Transform.from_xyz_rpy(xyz=(x, y, 0.), rpy=(0., 0., yaw))
        # Base pose lives in T_base; do not also apply the virtual base joints.
        torso_joint_dict = {
            "base_x": 0., "base_y": 0., "base_yaw": 0.,
            "torso_lift_joint": float(state.torso_q[0]),
        }
        return forward_kinematics_tree(
            self.tree, torso_joint_dict, base_transform=T_base,
        )["panda_link0"]

    def reset(self) -> InterpreterState:
        state = InterpreterState(
            arm_q={"main": self._home.copy()}, tcp_setpoint={},
            gripper_open={"main": 1.0}, step_m=0.02, rot_step_rad=np.pi / 12,
            base_se2=np.zeros(3), torso_q=np.zeros(1),
        )
        state.tcp_setpoint["main"] = self.fk_tcp(state, "main")
        return state

    def fk_tcp(self, state: InterpreterState, arm: str = "main") -> Transform:
        self.arm_joint_names(arm)
        return self.solver.fk(state.arm_q[arm], T_root=self._root(state))

    def apply_unit(
        self, unit: str, state: InterpreterState, arm: str = "main",
    ) -> tuple[InterpreterState, StepGroundResult]:
        self.arm_joint_names(arm)
        before = self.fk_tcp(state, arm).p.copy()

        def finish(ok=True, clamped=False, err=0.0, note=""):
            return state, StepGroundResult(
                unit=unit, arm=arm, accepted=bool(ok),
                clamped_by_workspace=bool(clamped), ik_pos_err_m=float(err),
                delta_tcp_world=self.fk_tcp(state, arm).p - before, note=note,
            )

        if unit in STEP_MODE_UNITS:
            state.step_m = float(STEP_MODE_UNITS[unit])
            return finish()
        if unit in ("GRASP", "RELEASE"):
            # Width = 0.08 * openness; modeled finger travel = 0.04 * openness.
            state.gripper_open[arm] = 0.0 if unit == "GRASP" else 1.0
            return finish()
        if unit in ("HOLD", "DONE"):
            return finish()
        if unit in MOBILE_BASE_UNITS and self.supports_mobile_base:
            dx, dy, dyaw = MOBILE_BASE_UNITS[unit]
            yaw = state.base_se2[2]
            c, s = np.cos(yaw), np.sin(yaw)
            raw = state.base_se2 + np.array([c * dx - s * dy, s * dx + c * dy, dyaw])
            candidate = raw.copy()
            lo = [self.tree.joints[n].lower for n in ("base_x", "base_y")]
            hi = [self.tree.joints[n].upper for n in ("base_x", "base_y")]
            candidate[:2] = np.clip(candidate[:2], lo, hi)
            candidate[2] = (candidate[2] + np.pi) % (2 * np.pi) - np.pi
            limited = not np.allclose(raw[:2], candidate[:2])
            state.base_se2 = candidate
            for name in self.arm_names:
                state.tcp_setpoint[name] = self.fk_tcp(state, name)
            return finish(note="Base translation limited by URDF bounds." if limited else "")
        if unit not in TRANSLATION_UNITS and unit not in ROTATION_UNITS:
            return finish(False, note=f"Unsupported unit: {unit}")

        current = state.tcp_setpoint[arm]
        clamped = False
        if unit in TRANSLATION_UNITS:
            p_raw = current.p + state.step_m * np.array(TRANSLATION_UNITS[unit])
            p = p_raw if self.supports_mobile_base else np.clip(p_raw, *self.workspace_bounds)
            clamped = not np.allclose(p_raw, p)
            target = Transform(R=current.R.copy(), p=p.copy())
        else:
            R_new = axis_angle_to_matrix(
                np.array(ROTATION_UNITS[unit], dtype=float), state.rot_step_rad,
            ) @ current.R
            target = Transform(R=R_new, p=current.p.copy())
        kwargs = dict(
            target_world=target, q_seed=state.arm_q[arm].copy(),
            T_root=self._root(state), pos_tol_m=1.5e-3, rot_tol_rad=2.5e-2,
            max_iters=120, num_restarts=8,
        )
        result = self.solver.solve_ik(**kwargs, rot_weight=1.0)
        accepted = result.success
        if not accepted and len(self._names) == 6:
            result = self.solver.solve_ik(**kwargs, rot_weight=0.15)
            accepted = result.pos_err_m <= 0.003
        if not accepted or not np.all(np.isfinite(result.q)):
            return finish(False, clamped, result.pos_err_m, "IK did not converge.")
        state.arm_q[arm] = result.q.copy()
        state.tcp_setpoint[arm] = self.fk_tcp(state, arm)
        return finish(True, clamped, result.pos_err_m)


def build_interpreter(urdf_path: str) -> EmbodimentInterpreter:
    return PandaOmronInterpreter(urdf_path)
