"""Exact tree forward kinematics, geometric Jacobians, manipulability, and damped-LS IK."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from eaisys.types import Transform, axis_angle_to_matrix
from eaisys.urdf.parser import JointNode, KinematicTree


def _joint_motion_transform(jt: JointNode, q_val: float) -> Transform:
    if jt.joint_type in ("revolute", "continuous"):
        return Transform(R=axis_angle_to_matrix(jt.axis, q_val), p=np.zeros(3, dtype=np.float64))
    if jt.joint_type == "prismatic":
        return Transform(R=np.eye(3, dtype=np.float64), p=jt.axis * float(q_val))
    return Transform.identity()


def forward_kinematics_tree(
    tree: KinematicTree,
    joint_values: dict[str, float],
    base_transform: Transform | None = None,
) -> dict[str, Transform]:
    """Compute world SE(3) transforms for every link in `tree`."""
    T_root = base_transform if base_transform is not None else Transform.identity()
    link_poses: dict[str, Transform] = {tree.root_link: T_root}
    stack = [tree.root_link]
    while stack:
        parent_link = stack.pop()
        T_parent = link_poses[parent_link]
        for jt in tree.children_map.get(parent_link, ()):
            if jt.mimic_joint is not None:
                raw_q = joint_values.get(jt.mimic_joint, 0.0) * jt.mimic_multiplier + jt.mimic_offset
            else:
                raw_q = joint_values.get(jt.name, 0.0)
            T_joint = jt.origin.compose(_joint_motion_transform(jt, raw_q))
            link_poses[jt.child] = T_parent.compose(T_joint)
            stack.append(jt.child)
    return link_poses


@dataclass
class IKResult:
    success: bool
    q: np.ndarray
    pos_err_m: float
    rot_err_rad: float
    iterations: int
    manipulability: float


class KinematicChainSolver:
    """Fast analytical FK, spatial Jacobian, and multi-restart damped-LS IK for one chain."""

    def __init__(
        self,
        tree: KinematicTree,
        root_link: str,
        tcp_link: str,
        actuated_joint_names: tuple[str, ...],
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        continuous_mask: np.ndarray,
        home_q: np.ndarray,
    ) -> None:
        self.tree = tree
        self.root_link = root_link
        self.tcp_link = tcp_link
        self.joint_path: list[JointNode] = tree.joint_path(root_link, tcp_link)
        self.actuated_joint_names = actuated_joint_names
        self.dof = len(actuated_joint_names)
        self.lower = np.asarray(lower_limits, dtype=np.float64)
        self.upper = np.asarray(upper_limits, dtype=np.float64)
        self.continuous_mask = np.asarray(continuous_mask, dtype=bool)
        self.home_q = np.asarray(home_q, dtype=np.float64)
        self._name_to_idx = {n: i for i, n in enumerate(actuated_joint_names)}

    def clamp_or_wrap(self, q: np.ndarray, q_ref: np.ndarray | None = None) -> np.ndarray:
        out = np.asarray(q, dtype=np.float64).copy()
        for i in range(self.dof):
            if self.continuous_mask[i]:
                ref = float(q_ref[i]) if q_ref is not None else 0.0
                out[i] = ref + ((out[i] - ref + math.pi) % (2.0 * math.pi) - math.pi)
            else:
                out[i] = float(np.clip(out[i], self.lower[i], self.upper[i]))
        return out

    def fk(self, q: np.ndarray, T_root: Transform | None = None) -> Transform:
        """Forward kinematics from `root_link` (at `T_root`) to `tcp_link`."""
        T = T_root if T_root is not None else Transform.identity()
        for jt in self.joint_path:
            idx = self._name_to_idx.get(jt.name)
            q_val = float(q[idx]) if idx is not None else 0.0
            T = T.compose(jt.origin).compose(_joint_motion_transform(jt, q_val))
        return T

    def fk_and_jacobian(
        self, q: np.ndarray, T_root: Transform | None = None
    ) -> tuple[Transform, np.ndarray]:
        """Compute TCP pose and 6xD geometric Jacobian [J_linear; J_angular] in world frame."""
        T = T_root if T_root is not None else Transform.identity()
        joint_origins_world: list[tuple[int, str, np.ndarray, np.ndarray]] = []
        for jt in self.joint_path:
            T_pre = T.compose(jt.origin)
            idx = self._name_to_idx.get(jt.name)
            if idx is not None:
                axis_world = T_pre.R @ jt.axis
                joint_origins_world.append((idx, jt.joint_type, T_pre.p.copy(), axis_world))
                T = T_pre.compose(_joint_motion_transform(jt, float(q[idx])))
            else:
                T = T_pre

        p_tcp = T.p
        J = np.zeros((6, self.dof), dtype=np.float64)
        for idx, jtype, p_j, z_j in joint_origins_world:
            if jtype in ("revolute", "continuous"):
                J[:3, idx] = np.cross(z_j, p_tcp - p_j)
                J[3:, idx] = z_j
            elif jtype == "prismatic":
                J[:3, idx] = z_j
        return T, J

    def manipulability(self, q: np.ndarray, T_root: Transform | None = None) -> float:
        _, J = self.fk_and_jacobian(q, T_root=T_root)
        JJt = J @ J.T
        det_val = float(np.linalg.det(JJt))
        return math.sqrt(max(0.0, det_val))

    def solve_ik(
        self,
        target_world: Transform,
        q_seed: np.ndarray | None = None,
        T_root: Transform | None = None,
        pos_tol_m: float = 1.5e-3,
        rot_tol_rad: float = 2.5e-2,
        rot_weight: float = 1.0,
        max_iters: int = 120,
        num_restarts: int = 8,
        rng_seed: int = 0,
    ) -> IKResult:
        """Damped least-squares IK with nullspace home-posture regularization and multi-restarts."""
        rng = np.random.default_rng(rng_seed)
        seeds: list[np.ndarray] = []
        if q_seed is not None:
            q0_clean = self.clamp_or_wrap(q_seed, q_ref=q_seed)
            seeds.append(q0_clean)
            for k in range(max(0, num_restarts - 1)):
                sigma = 0.08 + 0.06 * k
                perturbed = q0_clean + rng.normal(0.0, sigma, size=self.dof)
                seeds.append(self.clamp_or_wrap(perturbed, q_ref=q0_clean))
        else:
            seeds.append(self.clamp_or_wrap(self.home_q, q_ref=None))
            mid = 0.5 * (self.lower + self.upper)
            seeds.append(self.clamp_or_wrap(mid, q_ref=None))
            for _ in range(max(0, num_restarts - len(seeds))):
                lo_s = np.maximum(self.lower, -math.pi)
                hi_s = np.minimum(self.upper, math.pi)
                inverted = lo_s > hi_s
                lo_s = np.where(inverted, self.lower, lo_s)
                hi_s = np.where(inverted, self.upper, hi_s)
                sample = rng.uniform(lo_s, hi_s)
                seeds.append(self.clamp_or_wrap(sample, q_ref=None))

        best_res: IKResult | None = None
        weights = np.array([1.0, 1.0, 1.0, rot_weight, rot_weight, rot_weight], dtype=np.float64)
        W_diag = np.diag(weights)
        eye6 = np.eye(6, dtype=np.float64)
        eyeD = np.eye(self.dof, dtype=np.float64)

        for seed_idx, seed in enumerate(seeds):
            q = seed.copy()
            q_posture = (
                self.clamp_or_wrap(q_seed, q_ref=q_seed)
                if q_seed is not None
                else self.clamp_or_wrap(self.home_q, q_ref=q)
            )
            iters_used = 0
            for it in range(1, max_iters + 1):
                iters_used = it
                T_cur, J = self.fk_and_jacobian(q, T_root=T_root)
                err6 = T_cur.error_twist_world(target_world)
                pos_err = float(np.linalg.norm(err6[:3]))
                rot_err = float(np.linalg.norm(err6[3:]))
                if pos_err <= pos_tol_m and (rot_weight == 0.0 or rot_err <= rot_tol_rad):
                    manip = self.manipulability(q, T_root=T_root)
                    return IKResult(
                        success=True,
                        q=self.clamp_or_wrap(q, q_ref=q_seed),
                        pos_err_m=pos_err,
                        rot_err_rad=rot_err,
                        iterations=it,
                        manipulability=manip,
                    )

                err_w = W_diag @ err6
                err_norm = float(np.linalg.norm(err_w))
                if err_norm > 0.25:
                    err_w = err_w * (0.25 / err_norm)

                Jw = W_diag @ J
                damping = 1e-4 + 0.08 * min(1.0, err_norm)
                JJt = Jw @ Jw.T + (damping * damping) * eye6
                J_pinv = Jw.T @ np.linalg.solve(JJt, eye6)
                dq_task = J_pinv @ err_w

                # Nullspace bias toward comfortable posture on redundant (>=7-DoF) arms, scaled down near convergence
                if self.dof > 6:
                    null_proj = eyeD - J_pinv @ Jw
                    null_gain = 0.04 * min(1.0, err_norm / 0.02)
                    dq_null = null_gain * (q_posture - q)
                    dq = dq_task + null_proj @ dq_null
                else:
                    dq = dq_task

                max_step = float(np.max(np.abs(dq)))
                if max_step > 0.35:
                    dq = dq * (0.35 / max_step)
                q = self.clamp_or_wrap(q + dq, q_ref=q_seed if q_seed is not None else q)

            T_end = self.fk(q, T_root=T_root)
            err_end = T_end.error_twist_world(target_world)
            pos_err = float(np.linalg.norm(err_end[:3]))
            rot_err = float(np.linalg.norm(err_end[3:]))
            ok = pos_err <= pos_tol_m and (rot_weight == 0.0 or rot_err <= rot_tol_rad)
            cand = IKResult(
                success=ok,
                q=self.clamp_or_wrap(q, q_ref=q_seed),
                pos_err_m=pos_err,
                rot_err_rad=rot_err,
                iterations=iters_used,
                manipulability=self.manipulability(q, T_root=T_root),
            )
            if ok:
                return cand
            if best_res is None or (cand.pos_err_m + 0.2 * cand.rot_err_rad) < (
                best_res.pos_err_m + 0.2 * best_res.rot_err_rad
            ):
                best_res = cand

        assert best_res is not None
        return best_res
