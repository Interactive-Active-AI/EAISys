"""Kinematic Handshake: aligns URDF forward kinematics with recorded robot telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import numpy as np

from eaisys.types import Transform
from eaisys.urdf.kinematics import KinematicChainSolver


def _kabsch_rigid_alignment(src_pts: np.ndarray, dst_pts: np.ndarray) -> Transform:
    """Solve argmin_{R in SO(3), t} sum || R @ src_i + t - dst_i ||^2."""
    mu_s = src_pts.mean(axis=0)
    mu_d = dst_pts.mean(axis=0)
    S = src_pts - mu_s
    D = dst_pts - mu_d
    H = S.T @ D
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = mu_d - R @ mu_s
    return Transform(R=R, p=t)


@dataclass
class HandshakeReport:
    robot_id: str
    chain_name: str
    frame_convention: str  # "base_frame" | "home_relative"
    num_frames: int
    raw_pos_rmse_m: float
    aligned_pos_rmse_m: float
    aligned_rot_rmse_deg: float
    joint_signs: list[int]
    alignment_transform: Transform
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "chain_name": self.chain_name,
            "frame_convention": self.frame_convention,
            "num_frames": self.num_frames,
            "raw_pos_rmse_m": round(self.raw_pos_rmse_m, 5),
            "aligned_pos_rmse_m": round(self.aligned_pos_rmse_m, 5),
            "aligned_rot_rmse_deg": round(self.aligned_rot_rmse_deg, 3),
            "joint_signs": self.joint_signs,
            "alignment_transform": self.alignment_transform.to_dict(),
            "passed": self.passed,
        }


def kinematic_handshake(
    solver: KinematicChainSolver,
    robot_id: str,
    chain_name: str,
    q_traj: np.ndarray,
    recorded_eef_pose7_xyzw: np.ndarray,
    max_samples: int = 128,
    pass_tol_m: float = 0.025,
) -> HandshakeReport:
    """Verify and calibrate URDF FK against recorded EEF poses [x, y, z, qx, qy, qz, qw].

    Automatically handles both:
      1. `base_frame`: recorded EEF poses are expressed in a fixed base/world frame.
      2. `home_relative`: recorded EEF poses are expressed relative to the initial/q=0 TCP frame
         (as in RoboMIND 2.0 `end_effector_*_pose_align`).
    Also checks single-joint sign inversions when needed.
    """
    T_total = len(q_traj)
    if T_total == 0:
        raise ValueError("q_traj must contain at least one frame.")
    idx = np.linspace(0, T_total - 1, num=min(T_total, max_samples), dtype=int)
    q_sub = np.asarray(q_traj[idx], dtype=np.float64)
    rec_sub = np.asarray(recorded_eef_pose7_xyzw[idx], dtype=np.float64)

    rec_transforms = [Transform.from_xyz_quat_xyzw(r[:3], r[3:7]) for r in rec_sub]
    rec_pts = np.array([t.p for t in rec_transforms], dtype=np.float64)

    def _eval_signs(signs: np.ndarray) -> tuple[str, float, float, float, Transform]:
        fk_list = [solver.fk(q * signs) for q in q_sub]
        fk_pts_base = np.array([t.p for t in fk_list], dtype=np.float64)
        raw_rmse = float(np.sqrt(np.mean(np.sum((fk_pts_base - rec_pts) ** 2, axis=1))))

        # Mode A: fixed base alignment
        T_align_base = _kabsch_rigid_alignment(fk_pts_base, rec_pts)
        pred_base = (T_align_base.R @ fk_pts_base.T).T + T_align_base.p
        rmse_base = float(np.sqrt(np.mean(np.sum((pred_base - rec_pts) ** 2, axis=1))))

        # Mode B: relative-to-home TCP frame (T_home^{-1} @ T_fk(q_t))
        T_zero_inv = solver.fk(np.zeros(solver.dof, dtype=np.float64)).inverse()
        fk_rel_list = [T_zero_inv.compose(t) for t in fk_list]
        fk_pts_rel = np.array([t.p for t in fk_rel_list], dtype=np.float64)
        T_align_rel = _kabsch_rigid_alignment(fk_pts_rel, rec_pts)
        pred_rel = (T_align_rel.R @ fk_pts_rel.T).T + T_align_rel.p
        rmse_rel = float(np.sqrt(np.mean(np.sum((pred_rel - rec_pts) ** 2, axis=1))))

        start_near_origin = float(np.linalg.norm(rec_pts[0])) < 0.03
        if rmse_rel + 1e-6 < rmse_base or (
            start_near_origin and abs(rmse_rel - rmse_base) < 1e-3
        ):
            mode = "home_relative"
            best_rmse = rmse_rel
            T_align = T_align_rel
            src_tf = fk_rel_list
        else:
            mode = "base_frame"
            best_rmse = rmse_base
            T_align = T_align_base
            src_tf = fk_list

        # Compute orientation residual after estimating constant TCP frame rotation offset
        R_offsets = [s.R.T @ (T_align.R.T @ r.R) for s, r in zip(src_tf, rec_transforms)]
        R_mean = sum(R_offsets) / len(R_offsets)
        U, _, Vt = np.linalg.svd(R_mean)
        R_tcp_off = U @ Vt
        if np.linalg.det(R_tcp_off) < 0.0:
            U[:, -1] *= -1.0
            R_tcp_off = U @ Vt

        rot_errs_deg = []
        for s, r in zip(src_tf, rec_transforms):
            R_pred = T_align.R @ s.R @ R_tcp_off
            R_diff = r.R.T @ R_pred
            cos_a = max(-1.0, min(1.0, 0.5 * (float(np.trace(R_diff)) - 1.0)))
            rot_errs_deg.append(math.degrees(math.acos(cos_a)))
        rot_rmse_deg = float(np.sqrt(np.mean(np.square(rot_errs_deg))))
        return mode, raw_rmse, best_rmse, rot_rmse_deg, T_align

    signs = np.ones(solver.dof, dtype=np.float64)
    mode, raw_rmse, best_rmse, rot_rmse_deg, T_align = _eval_signs(signs)

    # Coordinate-descent sign check if initial residual exceeds tolerance
    if best_rmse > pass_tol_m:
        for j in range(solver.dof):
            trial = signs.copy()
            trial[j] *= -1.0
            m_t, raw_t, rmse_t, rot_t, T_t = _eval_signs(trial)
            if rmse_t + 1e-4 < best_rmse:
                signs = trial
                mode, raw_rmse, best_rmse, rot_rmse_deg, T_align = m_t, raw_t, rmse_t, rot_t, T_t

    return HandshakeReport(
        robot_id=robot_id,
        chain_name=chain_name,
        frame_convention=mode,
        num_frames=len(q_sub),
        raw_pos_rmse_m=raw_rmse,
        aligned_pos_rmse_m=best_rmse,
        aligned_rot_rmse_deg=rot_rmse_deg,
        joint_signs=[int(s) for s in signs],
        alignment_transform=T_align,
        passed=bool(best_rmse <= pass_tol_m),
    )
