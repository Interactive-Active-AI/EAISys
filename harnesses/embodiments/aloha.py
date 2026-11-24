import numpy as np
from eaisys.types import Transform, axis_angle_to_matrix
from eaisys.urdf import (
    KinematicChainSolver, forward_kinematics_tree, parse_robot_file,
)
from eaisys.harness.action_space import (
    ROTATION_UNITS, STEP_MODE_UNITS, TRANSLATION_UNITS,
)
from eaisys.harness.interpreter import (
    EmbodimentInterpreter, InterpreterState, StepGroundResult,
)


class AlohaInterpreter(EmbodimentInterpreter):
    robot_id = "aloha"
    arm_names = ("left", "right")
    supports_mobile_base = False
    max_gripper_width_m = 0.074

    def __init__(self, urdf_path: str):
        self.tree = parse_robot_file(urdf_path)
        suffixes = (
            "waist", "shoulder", "elbow", "forearm_roll",
            "wrist_angle", "wrist_rotate",
        )
        self._names = {
            a: tuple(f"{a}_{s}" for s in suffixes) for a in self.arm_names
        }
        self._home = np.array([0.0, -0.3, 0.6, 0.0, 0.5, 0.0])
        roots = forward_kinematics_tree(self.tree, {})
        self.solvers = {}
        for arm in self.arm_names:
            names = self._names[arm]
            nodes = [self.tree.joints[n] for n in names]
            lo = np.array([j.lower for j in nodes], dtype=float)
            hi = np.array([j.upper for j in nodes], dtype=float)
            if not np.all((self._home > lo) & (self._home < hi)):
                raise ValueError(f"{arm}: home must be strictly inside limits.")
            solver = KinematicChainSolver(
                tree=self.tree,
                root_link=f"{arm}_base_link",
                tcp_link=self.tcp_link_name(arm),
                actuated_joint_names=names,
                lower_limits=lo,
                upper_limits=hi,
                continuous_mask=np.array(
                    [j.is_continuous for j in nodes], dtype=bool,
                ),
                home_q=self._home.copy(),
            )
            quality = solver.manipulability(
                self._home, T_root=roots[f"{arm}_base_link"],
            )
            if not np.isfinite(quality) or quality <= 1e-6:
                raise ValueError(f"{arm}: sidecar home is poorly conditioned.")
            self.solvers[arm] = solver

        state = self.reset()
        points = np.array([state.tcp_setpoint[a].p for a in self.arm_names])
        # This home's forward reach permits a 0.105 m advance at fixed R.
        # The previous 0.4 m cap lay beyond reach: IK stopped before clipping.
        # Both mounts have identical x, so both arms share this reachable cap.
        padding = np.array([0.105, 0.105, 0.105])
        self.workspace_bounds = (
            points.min(axis=0) - padding, points.max(axis=0) + padding,
        )

    def arm_joint_names(self, arm: str = "left") -> tuple[str, ...]:
        return self._names[arm]

    def tcp_link_name(self, arm: str = "left") -> str:
        return f"{arm}_ee_tcp"

    def _root_transform(self, state, arm):
        x, y, yaw = state.base_se2
        T_base = Transform.from_xyz_rpy(
            xyz=(x, y, 0.0), rpy=(0.0, 0.0, yaw),
        )
        torso_joint_dict = {}  # This fixed-base URDF has no torso joints.
        return forward_kinematics_tree(
            self.tree, torso_joint_dict, base_transform=T_base,
        )[f"{arm}_base_link"]

    def reset(self) -> InterpreterState:
        state = InterpreterState(
            arm_q={a: self._home.copy() for a in self.arm_names},
            tcp_setpoint={},
            gripper_open={a: 1.0 for a in self.arm_names},
        )
        state.tcp_setpoint = {a: self.fk_tcp(state, a) for a in self.arm_names}
        return state

    def fk_tcp(self, state: InterpreterState, arm: str = "left") -> Transform:
        return self.solvers[arm].fk(
            state.arm_q[arm], T_root=self._root_transform(state, arm),
        )

    def apply_unit(self, unit: str, state: InterpreterState,
                   arm: str = "left") -> tuple[InterpreterState, StepGroundResult]:
        def finish(accepted, **kwargs):
            return state, StepGroundResult(
                unit=unit, arm=arm, accepted=bool(accepted), **kwargs,
            )

        if arm not in self.arm_names:
            return finish(False, note="Unknown arm.")
        if unit in STEP_MODE_UNITS:
            state.step_m = float(STEP_MODE_UNITS[unit])
            return finish(True)
        if unit in ("GRASP", "RELEASE"):
            state.gripper_open[arm] = 0.0 if unit == "GRASP" else 1.0
            return finish(True)
        if unit in ("HOLD", "DONE"):
            return finish(True)

        current = state.tcp_setpoint[arm]
        translation = unit in TRANSLATION_UNITS
        clamped = False
        if translation:
            p_raw = current.p + state.step_m * np.array(TRANSLATION_UNITS[unit])
            p_clamped = np.clip(p_raw, *self.workspace_bounds)
            clamped = not np.allclose(p_raw, p_clamped)
            target = Transform(R=current.R.copy(), p=p_clamped)
        elif unit in ROTATION_UNITS:
            R_new = axis_angle_to_matrix(
                np.array(ROTATION_UNITS[unit], dtype=float), state.rot_step_rad,
            ) @ current.R
            target = Transform(R=R_new, p=current.p.copy())
        else:
            return finish(False, note="Unsupported unit; the base is fixed.")

        solver = self.solvers[arm]
        kwargs = dict(
            target_world=target, q_seed=state.arm_q[arm].copy(),
            T_root=self._root_transform(state, arm),
            pos_tol_m=1.5e-3, rot_tol_rad=2.5e-2,
            max_iters=120, num_restarts=8,
        )
        result = solver.solve_ik(**kwargs, rot_weight=1.0)
        relaxed = translation and len(self._names[arm]) == 6 and not result.success
        if relaxed:
            result = solver.solve_ik(**kwargs, rot_weight=0.15)
        accepted = result.pos_err_m <= 0.003 if relaxed else result.success
        accepted = bool(accepted and np.all(np.isfinite(result.q)))
        delta = np.zeros(3)
        if accepted:
            before = self.fk_tcp(state, arm).p.copy()
            state.arm_q[arm] = result.q.copy()
            state.tcp_setpoint[arm] = self.fk_tcp(state, arm)
            delta = state.tcp_setpoint[arm].p - before
        note = ("Orientation relaxed." if relaxed else "") if accepted else "IK failed."
        return finish(
            accepted, clamped_by_workspace=clamped,
            ik_pos_err_m=float(result.pos_err_m), delta_tcp_world=delta, note=note,
        )


def build_interpreter(urdf_path: str) -> EmbodimentInterpreter:
    return AlohaInterpreter(urdf_path)
