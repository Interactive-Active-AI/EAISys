"""Reusable URDF grounding with world-frame arm actions and body-frame base actions."""

from dataclasses import dataclass
from itertools import product

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


@dataclass(frozen=True)
class ChainConfig:
    name: str
    root_link: str
    tcp_link: str
    joint_names: tuple[str, ...]
    home_q: list[float] | None = None


class GenericURDFInterpreter(EmbodimentInterpreter):
    """Automatic boxes follow mobile bases; explicit boxes remain world-fixed.

    Workspace boxes bound targets; IK still determines kinematic feasibility.
    """

    def __init__(
        self, urdf_path: str, robot_id: str, chains: list[ChainConfig],
        supports_mobile_base=False, torso_joints=(),
        max_gripper_width_m=0.08, workspace_bounds=None,
    ):
        self.tree = parse_robot_file(urdf_path)
        self.robot_id = robot_id
        self.arm_names = tuple(c.name for c in chains)
        if not chains or len(set(self.arm_names)) != len(chains):
            raise ValueError("Provide at least one chain, with unique chain names.")
        self._chains = {c.name: c for c in chains}
        self.supports_mobile_base = bool(supports_mobile_base)
        self.torso_joints = tuple(torso_joints)
        self.max_gripper_width_m = float(max_gripper_width_m)
        torso_nodes = [self.tree.joints[n] for n in self.torso_joints]
        self._torso_home = np.array([
            0.0 if j.is_continuous else float(np.clip(0.0, j.lower, j.upper))
            for j in torso_nodes
        ], dtype=float)
        self._homes, self.solvers = {}, {}
        self._workspace_corners = None
        rng = np.random.default_rng(0)

        for chain in chains:
            names = tuple(chain.joint_names)
            nodes = [self.tree.joints[n] for n in names]
            lower = np.array([j.lower for j in nodes], dtype=float)
            upper = np.array([j.upper for j in nodes], dtype=float)
            continuous = np.array([j.is_continuous for j in nodes], dtype=bool)
            automatic = chain.home_q is None
            if automatic:
                a = np.where(continuous, -2.0, np.maximum(lower, -2.0))
                b = np.where(continuous, 2.0, np.minimum(upper, 2.0))
                if np.any(a >= b) or not np.all(np.isfinite([a, b])):
                    raise ValueError(f"{chain.name}: empty home-sampling interval.")
                home = (a + b) / 2.0
            else:
                home = np.asarray(chain.home_q, dtype=float)
            if home.shape != (len(names),) or not np.all(np.isfinite(home)):
                raise ValueError(f"{chain.name}: invalid home configuration.")
            if np.any(((home <= lower) | (home >= upper)) & ~continuous):
                raise ValueError(f"{chain.name}: home must be strictly inside limits.")
            options = dict(
                tree=self.tree, root_link=chain.root_link, tcp_link=chain.tcp_link,
                actuated_joint_names=names, lower_limits=lower, upper_limits=upper,
                continuous_mask=continuous,
            )
            solver = KinematicChainSolver(**options, home_q=home.copy())
            if automatic:
                best, best_score = None, 1e-6
                margin = (b - a) * 1e-6
                for q in rng.uniform(a + margin, b - margin, size=(120, len(names))):
                    # Without T_root, FK and the forward-ready test are root-local.
                    p = solver.fk(q).p
                    if not np.all(np.isfinite(p)) or p[0] <= 0.28 or p[2] <= 0.14:
                        continue
                    quality = solver.manipulability(q)
                    if np.isfinite(quality) and quality > best_score:
                        best, best_score = q.copy(), float(quality)
                if best is None:
                    raise ValueError(
                        f"{chain.name}: no nonsingular forward-ready home sampled; "
                        "provide an explicit home_q."
                    )
                home = best
                # Give IK restarts the selected home through the documented API.
                solver = KinematicChainSolver(**options, home_q=home.copy())
            else:
                quality = solver.manipulability(home)
                if not np.isfinite(quality) or quality <= 1e-6:
                    raise ValueError(f"{chain.name}: home is poorly conditioned.")
            self._homes[chain.name] = home.copy()
            self.solvers[chain.name] = solver

        automatic_workspace = workspace_bounds is None
        state = self.reset()
        if automatic_workspace:
            points = np.array([state.tcp_setpoint[a].p for a in self.arm_names])
            workspace_bounds = (
                points.min(axis=0) - np.array([0.22, 0.35, 0.18]),
                points.max(axis=0) + np.array([0.28, 0.35, 0.25]),
            )
        lo, hi = (np.array(v, dtype=float, copy=True) for v in workspace_bounds)
        if (lo.shape != (3,) or hi.shape != (3,)
                or not np.all(np.isfinite([lo, hi])) or np.any(lo >= hi)):
            raise ValueError("Workspace bounds must be finite, ordered 3-vectors.")
        self.workspace_bounds = (lo, hi)
        if automatic_workspace and self.supports_mobile_base:
            self._workspace_corners = np.array(list(product(*zip(lo, hi))))

    def arm_joint_names(self, arm: str = "main") -> tuple[str, ...]:
        if arm not in self._chains:
            raise ValueError(f"Unknown arm: {arm}")
        return tuple(self._chains[arm].joint_names)

    def tcp_link_name(self, arm: str = "main") -> str:
        self.arm_joint_names(arm)
        return self._chains[arm].tcp_link

    @staticmethod
    def _base_transform(state: InterpreterState) -> Transform:
        x, y, yaw = state.base_se2
        return Transform.from_xyz_rpy(xyz=(x, y, 0.0), rpy=(0.0, 0.0, yaw))

    def _root(self, state: InterpreterState, arm: str) -> Transform:
        # Virtual base joints default to zero: apply base motion exactly once.
        return forward_kinematics_tree(
            self.tree, dict(zip(self.torso_joints, map(float, state.torso_q))),
            base_transform=self._base_transform(state),
        )[self._chains[arm].root_link]

    def _sync_workspace(self, state: InterpreterState):
        if self._workspace_corners is not None:
            T = self._base_transform(state)
            points = self._workspace_corners @ T.R.T + T.p
            self.workspace_bounds = (points.min(axis=0), points.max(axis=0))

    def reset(self) -> InterpreterState:
        state = InterpreterState(
            arm_q={a: self._homes[a].copy() for a in self.arm_names},
            tcp_setpoint={}, gripper_open={a: 1.0 for a in self.arm_names},
            torso_q=self._torso_home.copy(), base_se2=np.zeros(3),
            step_m=0.02, rot_step_rad=np.pi / 12,
        )
        state.tcp_setpoint = {a: self.fk_tcp(state, a) for a in self.arm_names}
        self._sync_workspace(state)
        return state

    def fk_tcp(self, state: InterpreterState, arm: str = "main") -> Transform:
        self.arm_joint_names(arm)
        return self.solvers[arm].fk(state.arm_q[arm], T_root=self._root(state, arm))

    def apply_unit(
        self, unit: str, state: InterpreterState, arm: str = "main",
    ) -> tuple[InterpreterState, StepGroundResult]:
        if arm not in self._chains:
            return state, StepGroundResult(
                unit=unit, arm=arm, accepted=False, note="Unknown arm.",
            )
        before = self.fk_tcp(state, arm)

        def finish(ok=True, clamped=False, err=0.0, note=""):
            return state, StepGroundResult(
                unit=unit, arm=arm, accepted=bool(ok),
                clamped_by_workspace=bool(clamped), ik_pos_err_m=float(err),
                delta_tcp_world=self.fk_tcp(state, arm).p - before.p, note=note,
            )

        if unit in STEP_MODE_UNITS:
            state.step_m = float(STEP_MODE_UNITS[unit])
            return finish()
        if unit in ("GRASP", "RELEASE"):
            state.gripper_open[arm] = float(unit == "RELEASE")
            return finish()
        if unit in ("HOLD", "DONE"):
            return finish()
        if unit in MOBILE_BASE_UNITS:
            if not self.supports_mobile_base:
                return finish(False, note="This robot has a fixed base.")
            dx, dy, dyaw = MOBILE_BASE_UNITS[unit]
            yaw = state.base_se2[2]
            c, s = np.cos(yaw), np.sin(yaw)
            raw = state.base_se2 + np.array([c * dx - s * dy, s * dx + c * dy, dyaw])
            candidate = raw.copy()
            # Honor conventional virtual-base limits when present in the URDF.
            for i, name in enumerate(("base_x", "base_y")):
                joint = self.tree.joints.get(name)
                if joint is not None and joint.joint_type == "prismatic":
                    candidate[i] = np.clip(candidate[i], joint.lower, joint.upper)
            limited = bool(np.any(candidate[:2] != raw[:2]))
            candidate[2] = (candidate[2] + np.pi) % (2 * np.pi) - np.pi
            state.base_se2 = candidate
            for name in self.arm_names:
                state.tcp_setpoint[name] = self.fk_tcp(state, name)
            self._sync_workspace(state)
            return finish(note="Base translation limited by URDF bounds." if limited else "")
        if unit not in TRANSLATION_UNITS and unit not in ROTATION_UNITS:
            return finish(False, note=f"Unsupported action unit: {unit}")

        self._sync_workspace(state)
        translation = unit in TRANSLATION_UNITS
        clamped = False
        if translation:
            raw = before.p + state.step_m * np.array(TRANSLATION_UNITS[unit])
            p = np.clip(raw, *self.workspace_bounds)
            clamped = bool(np.any(raw != p))
            target = Transform(R=before.R.copy(), p=p)
        else:
            R = axis_angle_to_matrix(
                np.array(ROTATION_UNITS[unit], dtype=float), state.rot_step_rad,
            ) @ before.R
            target = Transform(R=R, p=before.p.copy())
        solver = self.solvers[arm]
        options = dict(
            target_world=target, q_seed=state.arm_q[arm].copy(),
            T_root=self._root(state, arm), pos_tol_m=1.5e-3, rot_tol_rad=2.5e-2,
            max_iters=120, num_restarts=8,
        )
        result = solver.solve_ik(**options, rot_weight=1.0)
        relaxed = translation and len(self.arm_joint_names(arm)) == 6 and not result.success
        if relaxed:
            result = solver.solve_ik(**options, rot_weight=0.15)
        ok = result.pos_err_m <= 0.003 if relaxed else result.success
        if not ok or not np.all(np.isfinite(result.q)) or not np.isfinite(result.pos_err_m):
            return finish(False, clamped, result.pos_err_m, "IK target not reached.")
        state.arm_q[arm] = result.q.copy()
        state.tcp_setpoint[arm] = self.fk_tcp(state, arm)
        return finish(
            True, clamped, result.pos_err_m, "Orientation relaxed." if relaxed else "",
        )
