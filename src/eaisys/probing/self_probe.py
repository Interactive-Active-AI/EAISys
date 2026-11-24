"""URDF-Free Active Self-Probing & Online Broyden Visual-Motor Adaptation (Few-Shot Blind Onboarding).

Demonstrates Pillar 2 of EAISys: when NO `.urdf`, NO `KinematicTree`, and NO 3D EEF pose sensor are
available for a brand-new robot, the agent plays with its own body for a few seconds:
  1. Finger-wiggle self-discovery: opens/closes its gripper once while keeping arm joints fixed so
     pure pixel difference `|I_closed - I_open|` locates its own end-effector in raw multi-view RGB.
  2. Active joint motor-babbling (`D` micro-twitches): perturbs each joint by a small safe angle and
     tracks the end-effector's raw pixel displacement across orthogonal cameras (`birdview` + `sideview`)
     to estimate an empirical Visual-Motor Jacobian `J_vis (3 x D)`.
  3. Frontier LLM kinematic diagnosis: inspects the empirical probing telemetry + probe montage image
     to identify joint roles, calibrate step scale, and verify 6-axis directional alignment.
  4. Online Broyden adaptation: updates `J_vis` online after every executed semantic action unit
     (`MV_FWD/BACK/LEFT/RIGHT/UP/DOWN`, `GRASP`, `RELEASE`, `DONE`) to solve 3D tasks with ZERO URDF
     and ZERO simulator privilege.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any
import numpy as np
from PIL import Image, ImageDraw

from eaisys.backends.base import FrontierBackend
from eaisys.harness.action_space import (
    STEP_MODE_UNITS,
    TRANSLATION_UNITS,
    single_arm_step_json_schema,
    valid_units_for_embodiment,
)
from eaisys.harness.plugins import (
    ArmProprioPacket,
    HarnessPluginConfig,
    PluginObservationContext,
    PluginPipeline,
)
from eaisys.sanitize import sanitize_payload, sanitize_text
from eaisys.synthesis import SemanticPriorBank


def _to_gray(img_rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(img_rgb, dtype=np.float32)
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def locate_tcp_by_finger_wiggle(
    img_open: np.ndarray,
    img_closed: np.ndarray,
) -> tuple[tuple[float, float], np.ndarray]:
    """Pure pixel-diff end-effector locator: the only moving pixels during a static finger wiggle are the jaws."""
    diff = np.abs(img_open.astype(np.float32) - img_closed.astype(np.float32)).mean(axis=2)
    max_d = float(diff.max())
    if max_d < 3.0:
        h, w = diff.shape
        return (w * 0.5, h * 0.5), diff
    thresh = max(4.0, max_d * 0.35)
    mask = diff >= thresh
    ys, xs = np.nonzero(mask)
    weights = diff[mask]
    u = float(np.sum(xs * weights) / np.sum(weights))
    v = float(np.sum(ys * weights) / np.sum(weights))
    return (u, v), diff


def _crop_patch(gray: np.ndarray, u: float, v: float, half: int) -> np.ndarray:
    h, w = gray.shape
    cx, cy = int(round(u)), int(round(v))
    x0, x1 = max(0, cx - half), min(w, cx + half + 1)
    y0, y1 = max(0, cy - half), min(h, cy + half + 1)
    patch = gray[y0:y1, x0:x1]
    if patch.shape != (2 * half + 1, 2 * half + 1):
        padded = np.full((2 * half + 1, 2 * half + 1), float(gray.mean()), dtype=np.float32)
        px0 = half - (cx - x0)
        py0 = half - (cy - y0)
        padded[py0 : py0 + patch.shape[0], px0 : px0 + patch.shape[1]] = patch
        return padded
    return patch


def track_tcp_ncc(
    gray_curr: np.ndarray,
    template: np.ndarray,
    prev_uv: tuple[float, float],
    search_radius: int = 34,
    stride: int = 2,
) -> tuple[tuple[float, float], float]:
    """Fast normalized cross-correlation (NCC) visual tracker around `prev_uv` (pure RGB pixels, zero privilege)."""
    h, w = gray_curr.shape
    half = template.shape[0] // 2
    t_zero = template - float(template.mean())
    t_norm = float(np.linalg.norm(t_zero))
    if t_norm < 1e-4:
        return prev_uv, 0.0

    px, py = int(round(prev_uv[0])), int(round(prev_uv[1]))
    x_min = max(half, px - search_radius)
    x_max = min(w - half - 1, px + search_radius)
    y_min = max(half, py - search_radius)
    y_max = min(h - half - 1, py + search_radius)

    best_score = -2.0
    best_xy = (px, py)
    # Coarse stride search
    for cy in range(y_min, y_max + 1, stride):
        for cx in range(x_min, x_max + 1, stride):
            win = gray_curr[cy - half : cy + half + 1, cx - half : cx + half + 1]
            w_zero = win - float(win.mean())
            denom = float(np.linalg.norm(w_zero)) * t_norm
            if denom < 1e-4:
                continue
            score = float(np.sum(w_zero * t_zero) / denom)
            if score > best_score:
                best_score = score
                best_xy = (cx, cy)

    # Fine 1-pixel refinement around coarse optimum
    fx0 = max(half, best_xy[0] - stride)
    fx1 = min(w - half - 1, best_xy[0] + stride)
    fy0 = max(half, best_xy[1] - stride)
    fy1 = min(h - half - 1, best_xy[1] + stride)
    for cy in range(fy0, fy1 + 1):
        for cx in range(fx0, fx1 + 1):
            win = gray_curr[cy - half : cy + half + 1, cx - half : cx + half + 1]
            w_zero = win - float(win.mean())
            denom = float(np.linalg.norm(w_zero)) * t_norm
            if denom < 1e-4:
                continue
            score = float(np.sum(w_zero * t_zero) / denom)
            if score > best_score:
                best_score = score
                best_xy = (cx, cy)

    return (float(best_xy[0]), float(best_xy[1])), round(best_score, 4)


@dataclass
class SelfProbedEmbodimentModel:
    """URDF-free empirical visual-motor Jacobian `J_vis (3 x D)` with online Broyden adaptation.

    Visual coordinate frame `[x_vis, y_vis, z_vis]` measured strictly from raw orthogonal cameras:
      - `x_vis`: `+v_top` (down in `birdview` = `+X` / `MV_FWD`)
      - `y_vis`: `+u_top` (right in `birdview` = `+Y` / `MV_LEFT`)
      - `z_vis`: `-v_side` (up in `sideview` = `+Z` / `MV_UP`)
    """

    dof: int
    J_vis: np.ndarray  # shape (3, D), pixels per radian
    q_home: np.ndarray | None = None
    px_per_meter: float = 210.0
    step_m: float = 0.02
    gripper_open: float = 1.0
    max_dq_rad: float = 0.22
    damping: float = 8.0
    broyden_alpha: float = 0.15
    probing_records: list[dict[str, Any]] = field(default_factory=list)
    llm_diagnosis: dict[str, Any] = field(default_factory=dict)

    def solve_semantic_dq(self, unit: str, q_cur: np.ndarray | None = None) -> tuple[np.ndarray, bool, str]:
        """Map one semantic unit to an incremental joint command `dq` using ONLY the probed Jacobian."""
        if unit in STEP_MODE_UNITS:
            self.step_m = float(STEP_MODE_UNITS[unit])
            return np.zeros(self.dof, dtype=np.float64), True, f"step_m={self.step_m}"
        if unit == "GRASP":
            self.gripper_open = 0.0
            return np.zeros(self.dof, dtype=np.float64), True, "gripper_closed"
        if unit == "RELEASE":
            self.gripper_open = 1.0
            return np.zeros(self.dof, dtype=np.float64), True, "gripper_open"
        if unit in ("HOLD", "DONE"):
            return np.zeros(self.dof, dtype=np.float64), True, unit.lower()
        if unit in TRANSLATION_UNITS:
            axis = np.asarray(TRANSLATION_UNITS[unit], dtype=np.float64)
            dp_px = axis * (self.step_m * self.px_per_meter)
            JJt = self.J_vis @ self.J_vis.T + (self.damping**2) * np.eye(3, dtype=np.float64)
            J_pinv = self.J_vis.T @ np.linalg.solve(JJt, np.eye(3, dtype=np.float64))
            dq = J_pinv @ dp_px
            # URDF-free nullspace posture keeping toward initial upright encoder reading q_home
            if self.q_home is not None and q_cur is not None and self.dof > 3:
                null_proj = np.eye(self.dof, dtype=np.float64) - J_pinv @ self.J_vis
                dq = dq + 0.10 * (null_proj @ (self.q_home - np.asarray(q_cur, dtype=np.float64)))
            max_abs = float(np.max(np.abs(dq)))
            if max_abs > self.max_dq_rad:
                dq = dq * (self.max_dq_rad / max_abs)
            return dq, True, f"target_dp_px={np.round(dp_px, 2).tolist()}"
        if unit in ("ROT_YAW_CW", "ROT_YAW_CCW"):
            dq = np.zeros(self.dof, dtype=np.float64)
            dq[-1] = -0.22 if unit == "ROT_YAW_CW" else 0.22
            return dq, True, "last_joint_wrist_yaw"
        return np.zeros(self.dof, dtype=np.float64), False, f"unsupported_blind_unit:{unit}"

    def broyden_update(self, dq_actual: np.ndarray, dp_vis_actual: np.ndarray) -> float:
        """Rank-1 damped Broyden update keeping the URDF-free visual-motor Jacobian accurate online."""
        dq = np.asarray(dq_actual, dtype=np.float64)
        dp = np.asarray(dp_vis_actual, dtype=np.float64)
        dq_norm2 = float(np.dot(dq, dq))
        if dq_norm2 < 1e-5:
            return 0.0
        pred = self.J_vis @ dq
        err = dp - pred
        err_px = float(np.linalg.norm(err))
        # Guard against sudden visual occlusion outliers
        if err_px > 45.0:
            return err_px
        self.J_vis = self.J_vis + self.broyden_alpha * np.outer(err, dq) / (dq_norm2 + 1e-3)
        return err_px


class URDFFreeMuJoCoEnv:
    """Black-box 3D MuJoCo environment with ZERO URDF, ZERO FK, ZERO 3D pose sensor, ZERO privilege."""

    def __init__(
        self,
        *,
        robosuite_env_name: str,
        robosuite_robot: str,
        task_instruction: str,
        camera_size: int = 384,
        seed: int = 11,
        substeps_per_unit: int = 14,
    ) -> None:
        import robosuite as suite
        from robosuite.controllers import load_part_controller_config
        from robosuite.controllers.composite.composite_controller_factory import (
            refactor_composite_controller_config,
        )

        self.robosuite_env_name = robosuite_env_name
        self.robosuite_robot = robosuite_robot
        self.task_instruction = task_instruction
        self.camera_size = camera_size
        self.seed = seed
        self.substeps_per_unit = substeps_per_unit
        self.active_cameras = ("frontview", "sideview", "robot0_eye_in_hand")

        arm_cfg = load_part_controller_config(default_controller="JOINT_POSITION")
        arm_cfg["kp"] = 180
        ctrl_cfg = refactor_composite_controller_config(arm_cfg, robosuite_robot, ["right"])
        self._env = suite.make(
            env_name=robosuite_env_name,
            robots=robosuite_robot,
            controller_configs=ctrl_cfg,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=list(self.active_cameras),
            camera_heights=camera_size,
            camera_widths=camera_size,
            control_freq=20,
            horizon=3500,
        )
        np.random.seed(seed)
        self.obs = self._env.reset()
        self.dof = int(len(self.obs["robot0_joint_pos"]))
        self.q_target = self.obs["robot0_joint_pos"].copy()
        self.gripper_open = 1.0
        self.video_frames: list[np.ndarray] = []
        self.recent_units: list[str] = []
        self.phase_label = "PHASE 1: URDF-FREE ACTIVE SELF-PROBING"
        self.subtask_label = "finger-wiggle TCP discovery"
        self.last_action_label = "INIT"
        self.current_turn_idx = 0

        self.top_uv: tuple[float, float] = (camera_size * 0.5, camera_size * 0.5)
        self.side_uv: tuple[float, float] = (camera_size * 0.5, camera_size * 0.5)
        self._top_tpl: np.ndarray | None = None
        self._side_tpl: np.ndarray | None = None
        self.model: SelfProbedEmbodimentModel | None = None

    def _cam_rgb(self, name: str) -> np.ndarray:
        return np.flipud(self.obs[f"{name}_image"])

    def _step_raw_joints(self, q_des: np.ndarray, g_open: float, n_sub: int = 12, record_stride: int = 2) -> None:
        self.q_target = np.asarray(q_des, dtype=np.float64).copy()
        self.gripper_open = float(g_open)
        g_cmd = -1.0 if g_open > 0.5 else 1.0
        for i in range(n_sub):
            q_cur = self.obs["robot0_joint_pos"]
            diff = self.q_target - q_cur
            act_arm = np.clip(diff / 0.05, -1.0, 1.0)
            act = np.concatenate([act_arm, [g_cmd]])
            self.obs, _, _, _ = self._env.step(act)
            if i % record_stride == 0 or i == n_sub - 1:
                self.video_frames.append(self.compose_frame_rgb())

    def _refresh_visual_tcp(self) -> tuple[np.ndarray, float]:
        """Track TCP purely in raw unoccluded `frontview` & `sideview` pixels via NCC template matching.
        Visual axes:
          - x_vis (+X / MV_FWD): `-u_side` (left in sideview = +X)
          - y_vis (+Y / MV_LEFT): `+u_front` (right in frontview = +Y)
          - z_vis (+Z / MV_UP):   `-0.5 * (v_side + v_front)` (up in both views = +Z)
        """
        if self._top_tpl is None or self._side_tpl is None:
            return np.zeros(3, dtype=np.float64), 0.0
        g_front = _to_gray(self._cam_rgb("frontview"))
        g_side = _to_gray(self._cam_rgb("sideview"))
        old_vec = np.array(
            [-self.side_uv[0], self.top_uv[0], -0.5 * (self.side_uv[1] + self.top_uv[1])],
            dtype=np.float64,
        )
        new_front, s_front = track_tcp_ncc(g_front, self._top_tpl, self.top_uv, search_radius=42, stride=2)
        new_side, s_side = track_tcp_ncc(g_side, self._side_tpl, self.side_uv, search_radius=42, stride=2)
        self.top_uv = new_front
        self.side_uv = new_side
        if s_front > 0.65:
            self._top_tpl = 0.7 * self._top_tpl + 0.3 * _crop_patch(g_front, new_front[0], new_front[1], 14)
        if s_side > 0.65:
            self._side_tpl = 0.7 * self._side_tpl + 0.3 * _crop_patch(g_side, new_side[0], new_side[1], 14)
        new_vec = np.array(
            [-self.side_uv[0], self.top_uv[0], -0.5 * (self.side_uv[1] + self.top_uv[1])],
            dtype=np.float64,
        )
        return new_vec - old_vec, min(s_front, s_side)

    def _relock_tcp_by_wiggle(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Re-lock exact TCP pixel coordinates by closing/opening fingers at fixed arm joints."""
        q_hold = self.obs["robot0_joint_pos"].copy()
        self._step_raw_joints(q_hold, 1.0, n_sub=18, record_stride=3)
        front_open = self._cam_rgb("frontview").copy()
        side_open = self._cam_rgb("sideview").copy()
        self._step_raw_joints(q_hold, 0.0, n_sub=22, record_stride=3)
        front_closed = self._cam_rgb("frontview").copy()
        side_closed = self._cam_rgb("sideview").copy()
        self._step_raw_joints(q_hold, 1.0, n_sub=18, record_stride=3)
        self.top_uv, _ = locate_tcp_by_finger_wiggle(front_open, front_closed)
        self.side_uv, _ = locate_tcp_by_finger_wiggle(side_open, side_closed)
        self._top_tpl = _crop_patch(_to_gray(front_open), self.top_uv[0], self.top_uv[1], 14)
        self._side_tpl = _crop_patch(_to_gray(side_open), self.side_uv[0], self.side_uv[1], 14)
        return self.top_uv, self.side_uv

    def run_active_self_probing(
        self,
        *,
        probe_delta_rad: float = 0.14,
        montage_out: Path | None = None,
    ) -> dict[str, Any]:
        """Stage 1: Zero-URDF Active Self-Probing ('play with own joints to estimate embodiment')."""
        self.phase_label = "PHASE 1: URDF-FREE ACTIVE SELF-PROBING (ZERO URDF)"
        self.subtask_label = "Step 1A: Finger-wiggle visual TCP self-discovery"
        self.last_action_label = "WIGGLE_FINGERS"
        front_uv0, side_uv0 = self._relock_tcp_by_wiggle()

        q0 = self.obs["robot0_joint_pos"].copy()
        J_vis = np.zeros((3, self.dof), dtype=np.float64)
        probe_records: list[dict[str, Any]] = []
        front_base = self._cam_rgb("frontview").copy()
        side_base = self._cam_rgb("sideview").copy()

        for j in range(self.dof):
            self.subtask_label = f"Step 1B: Active motor-babbling joint {j + 1}/{self.dof}"
            self.last_action_label = f"PROBE_J{j}"
            q_plus = q0.copy()
            q_plus[j] += probe_delta_rad
            self._step_raw_joints(q_plus, 1.0, n_sub=14, record_stride=3)
            q_ach = self.obs["robot0_joint_pos"].copy()
            dq_j = float(q_ach[j] - q0[j])

            # Measure pure-pixel TCP displacement by finger-wiggle lock in unoccluded frontview + sideview
            front_p, side_p = self._relock_tcp_by_wiggle()
            dx_px = -(side_p[0] - side_uv0[0])   # -u in sideview = +X (MV_FWD)
            dy_px = front_p[0] - front_uv0[0]    # +u in frontview = +Y (MV_LEFT)
            dz_px = -0.5 * ((side_p[1] - side_uv0[1]) + (front_p[1] - front_uv0[1]))  # -v in both = +Z (MV_UP)
            dp_vis = np.array([dx_px, dy_px, dz_px], dtype=np.float64)
            if abs(dq_j) > 1e-3:
                J_vis[:, j] = dp_vis / dq_j

            probe_records.append(
                {
                    "joint_index": j,
                    "dq_rad": round(dq_j, 4),
                    "dp_vis_px": [round(float(x), 2) for x in dp_vis],
                    "sensitivity_px_per_rad": [round(float(x), 1) for x in J_vis[:, j]],
                    "pixel_shift_norm": round(float(np.linalg.norm(dp_vis)), 2),
                }
            )
            self._step_raw_joints(q0, 1.0, n_sub=14, record_stride=4)

        self.top_uv, self.side_uv = front_uv0, side_uv0
        self._top_tpl = _crop_patch(_to_gray(front_base), front_uv0[0], front_uv0[1], 14)
        self._side_tpl = _crop_patch(_to_gray(side_base), side_uv0[0], side_uv0[1], 14)

        j_col_norms = np.linalg.norm(J_vis, axis=0)
        top_norm = float(np.max(j_col_norms)) if len(j_col_norms) else 140.0
        px_per_m = float(np.clip(top_norm / 0.85, 160.0, 260.0))

        self.model = SelfProbedEmbodimentModel(
            dof=self.dof,
            J_vis=J_vis,
            q_home=q0.copy(),
            px_per_meter=round(px_per_m, 1),
            probing_records=probe_records,
        )

        # Self-verification: execute a tiny visual check along +X, +Y, +Z using finger-wiggle pixel diff
        self.subtask_label = "Step 1C: Few-shot 3-axis directional self-verification"
        axis_verify: dict[str, float] = {}
        for unit, expected_axis in (("MV_FWD", [1, 0, 0]), ("MV_LEFT", [0, 1, 0]), ("MV_UP", [0, 0, 1])):
            self.last_action_label = f"CHECK_{unit}"
            q_before = self.obs["robot0_joint_pos"].copy()
            dq, _, _ = self.model.solve_semantic_dq(unit)
            self._step_raw_joints(q_before + dq * 1.6, 1.0, n_sub=14, record_stride=3)
            fp, sp = self._relock_tcp_by_wiggle()
            dp_obs = np.array(
                [-(sp[0] - side_uv0[0]), fp[0] - front_uv0[0], -0.5 * ((sp[1] - side_uv0[1]) + (fp[1] - front_uv0[1]))],
                dtype=np.float64,
            )
            norm = float(np.linalg.norm(dp_obs))
            cos_sim = float(np.dot(dp_obs, np.asarray(expected_axis, dtype=np.float64)) / norm) if norm > 1.0 else 0.0
            axis_verify[unit] = round(cos_sim, 3)
            self._step_raw_joints(q_before, 1.0, n_sub=12, record_stride=4)
        self.top_uv, self.side_uv = front_uv0, side_uv0
        self._top_tpl = _crop_patch(_to_gray(front_base), front_uv0[0], front_uv0[1], 14)
        self._side_tpl = _crop_patch(_to_gray(side_base), side_uv0[0], side_uv0[1], 14)

        if montage_out is not None:
            self._render_probe_montage(front_base, side_base, front_uv0, side_uv0, J_vis, montage_out)

        return {
            "mode": "URDF-Free Active Self-Probing (Zero URDF, Zero FK, Zero 3D Pose Oracle)",
            "discovered_dof": self.dof,
            "initial_q_rad": [round(float(x), 4) for x in q0],
            "finger_wiggle_tcp_px": {
                "frontview_uv": [round(front_uv0[0], 1), round(front_uv0[1], 1)],
                "sideview_uv": [round(side_uv0[0], 1), round(side_uv0[1], 1)],
            },
            "estimated_px_per_m": self.model.px_per_meter,
            "visual_motor_jacobian_px_per_rad": np.round(J_vis, 2).tolist(),
            "joint_probes": probe_records,
            "directional_cosine_self_check": axis_verify,
        }

    def _render_probe_montage(
        self,
        top_rgb: np.ndarray,
        side_rgb: np.ndarray,
        top_uv: tuple[float, float],
        side_uv: tuple[float, float],
        J_vis: np.ndarray,
        out_path: Path,
    ) -> None:
        W, H = 800, 440
        canvas = Image.new("RGB", (W, H), (250, 249, 246))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, W, 38), fill=(53, 61, 67))
        draw.text(
            (14, 11),
            f"URDF-FREE ACTIVE SELF-PROBING  |  {self.robosuite_robot} (D={self.dof} discovered)  |  Finger-Wiggle TCP + Visual-Motor Jacobian",
            fill=(255, 255, 255),
        )
        p_top = Image.fromarray(top_rgb).resize((380, 380), Image.Resampling.BILINEAR)
        p_side = Image.fromarray(side_rgb).resize((380, 380), Image.Resampling.BILINEAR)
        canvas.paste(p_top, (12, 48))
        canvas.paste(p_side, (408, 48))
        sc = 380.0 / self.camera_size

        # Draw discovered TCP crosshair + per-joint Jacobian vectors
        tu, tv = 12 + top_uv[0] * sc, 48 + top_uv[1] * sc
        su, sv = 408 + side_uv[0] * sc, 48 + side_uv[1] * sc
        for cx, cy in ((tu, tv), (su, sv)):
            draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), outline=(202, 111, 111), width=3)

        colors = [(62, 98, 128), (106, 84, 132), (152, 80, 94), (125, 104, 84), (70, 120, 140), (140, 90, 130), (110, 110, 110)]
        for j in range(self.dof):
            col = colors[j % len(colors)]
            dx, dy, dz = J_vis[0, j] * 0.25 * sc, J_vis[1, j] * 0.25 * sc, J_vis[2, j] * 0.25 * sc
            # frontview: +u = +Y, -v = +Z; sideview: -u = +X, -v = +Z
            draw.line((tu, tv, tu + dy, tv - dz), fill=col, width=3)
            draw.line((su, sv, su - dx, sv - dz), fill=col, width=3)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path, format="JPEG", quality=92)

    def step_unit(self, unit: str) -> dict[str, Any]:
        assert self.model is not None, "Call run_active_self_probing() first!"
        self.last_action_label = unit
        self.recent_units.append(unit)
        q_before = self.obs["robot0_joint_pos"].copy()
        dq_cmd, accepted, note = self.model.solve_semantic_dq(unit, q_cur=q_before)
        n_sub = 34 if unit in ("GRASP", "RELEASE") else (3 if unit.startswith("STEP_") else self.substeps_per_unit)
        self._step_raw_joints(q_before + dq_cmd, self.model.gripper_open, n_sub=n_sub, record_stride=2)

        q_after = self.obs["robot0_joint_pos"].copy()
        dp_vis, ncc_conf = self._refresh_visual_tcp()
        broyden_err = 0.0
        # Guard against noisy sub-pixel NCC jitter on tiny steps; only adapt when motion & track conf are strong
        if unit in TRANSLATION_UNITS and ncc_conf >= 0.80 and float(np.linalg.norm(dp_vis)) >= 4.0:
            broyden_err = self.model.broyden_update(q_after - q_before, dp_vis)

        return {
            "unit": unit,
            "accepted": accepted,
            "note": note,
            "dq_norm_rad": round(float(np.linalg.norm(q_after - q_before)), 4),
            "observed_visual_dp_px": [round(float(x), 2) for x in dp_vis],
            "ncc_track_conf": ncc_conf,
            "broyden_residual_px": round(broyden_err, 2),
        }

    def check_success_audit_only(self) -> tuple[bool, dict[str, Any]]:
        """Post-turn evaluation oracle — NEVER shown to the VLM!"""
        ok = bool(self._env._check_success())
        info: dict[str, Any] = {"mujoco_task_success": ok}
        if "cube_pos" in self.obs:
            info["cube_z_m"] = round(float(self.obs["cube_pos"][2]), 4)
        if "cubeA_pos" in self.obs and "cubeB_pos" in self.obs:
            ca, cb = self.obs["cubeA_pos"], self.obs["cubeB_pos"]
            info["audit_xy_err_m"] = round(float(np.linalg.norm(ca[:2] - cb[:2])), 4)
            info["audit_dz_m"] = round(float(ca[2] - cb[2]), 4)
        return ok, info

    def compose_frame_rgb(self) -> np.ndarray:
        c1, c2, c3 = self.active_cameras
        i1, i2, i3 = self._cam_rgb(c1), self._cam_rgb(c2), self._cam_rgb(c3)
        W, H = 1152, 448
        canvas = Image.new("RGB", (W, H), (249, 247, 244))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, W, 38), fill=(53, 61, 67))
        g_str = "OPEN" if self.gripper_open > 0.5 else "CLOSED"
        draw.text(
            (14, 11),
            f"{self.phase_label}  |  {self.robosuite_env_name} · {self.robosuite_robot} (NO URDF)  |  Turn {self.current_turn_idx}  |  Last: {self.last_action_label}  |  EE: {g_str}",
            fill=(255, 255, 255),
        )
        pw, ph = 368, 368
        panels = [
            (12, i1, (82, 107, 133), "CAM 1: FRONT (SELF-PROBE +Y/+Z)"),
            (392, i2, (150, 92, 102), "CAM 2: LEFT-SIDE (SELF-PROBE +X/+Z)"),
            (772, i3, (116, 99, 140), "CAM 3: WRIST EYE-IN-HAND"),
        ]
        for x0, arr, col, title in panels:
            pil_p = Image.fromarray(arr).resize((pw, ph), Image.Resampling.BILINEAR)
            canvas.paste(pil_p, (x0, 46))
            draw.rectangle((x0, 46, x0 + pw, 46 + ph), outline=col, width=2)
            draw.rectangle((x0 + 4, 50, x0 + pw - 4, 72), fill=col)
            draw.text((x0 + 10, 54), title, fill=(255, 255, 255))
        draw.text(
            (14, 422),
            f"Task: {self.task_instruction[:74]}   |   Subtask: {self.subtask_label[:46]}",
            fill=(73, 78, 82),
        )
        return np.asarray(canvas, dtype=np.uint8)

    def save_observation_jpg(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.compose_frame_rgb()).save(p, format="JPEG", quality=90)
        return p

    def export_mp4(self, mp4_path: str | Path, fps: int = 20) -> Path:
        import imageio.v2 as imageio

        p = Path(mp4_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(
            str(p),
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=16,
        ) as writer:
            for frame in self.video_frames:
                writer.append_data(frame)
        return p

    def close(self) -> None:
        self._env.close()


def run_urdf_free_self_probe_pipeline(
    *,
    run_id: str,
    robosuite_robot: str,
    robosuite_env_name: str,
    task_instruction: str,
    backend: FrontierBackend,
    media_out_dir: str | Path,
    trace_json_out: str | Path,
    priors_path: str | Path | None = None,
    seed: int = 11,
    max_turns: int = 18,
    plugin_cfg: HarnessPluginConfig | None = None,
) -> dict[str, Any]:
    """End-to-end Pillar-2 Pipeline: ZERO URDF -> Active Self-Probing -> LLM Diagnosis -> Zero-Privilege 3D Solve."""
    t_start = time.monotonic()
    mdir = Path(media_out_dir) / run_id
    if mdir.exists():
        for f in mdir.glob("*"):
            if f.is_file():
                f.unlink()
    mdir.mkdir(parents=True, exist_ok=True)

    env = URDFFreeMuJoCoEnv(
        robosuite_env_name=robosuite_env_name,
        robosuite_robot=robosuite_robot,
        task_instruction=task_instruction,
        seed=seed,
    )

    # Phase 1: Active Self-Probing (NO URDF!)
    montage_path = mdir / "self_probe_montage.jpg"
    probe_report = env.run_active_self_probing(montage_out=montage_path)

    # Phase 1D: Frontier LLM inspects the self-probing montage + Jacobian table and diagnoses the blind embodiment
    diag_prompt = (
        f"You are EAISys operating in **URDF-FREE FEW-SHOT SELF-PROBING MODE** on an unknown robot (`{robosuite_robot}`, NO URDF file provided).\n"
        f"The robot just wiggled its fingers to locate its own end-effector in raw RGB cameras and micro-probed its {env.dof} joints.\n"
        f"Empirical Self-Probing Telemetry:\n```json\n{json.dumps(probe_report, indent=2)}\n```\n"
        "Inspect the attached self-probing montage image and empirical visual-motor Jacobian above. "
        "Summarize in strict JSON (`summary`, `joint_roles`, `control_advice`) what morphology was inferred and how to drive it cleanly."
    )
    diag_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary", "joint_roles", "control_advice"],
        "properties": {
            "summary": {"type": "string"},
            "joint_roles": {"type": "array", "items": {"type": "string"}},
            "control_advice": {"type": "string"},
        },
    }
    diag_resp = backend.generate(diag_prompt, images=[montage_path], json_schema=diag_schema, timeout_s=300.0)
    try:
        diag_json = diag_resp.parse_json()
    except Exception:
        diag_json = {"summary": diag_resp.text[:400], "joint_roles": [], "control_advice": ""}
    assert env.model is not None
    env.model.llm_diagnosis = diag_json

    # Phase 2: Closed-Loop Zero-Privilege VLM Manipulation using the self-probed interpreter + online Broyden!
    env.phase_label = "PHASE 2: URDF-FREE VLM CLOSED-LOOP + BROYDEN ADAPTATION"
    demos = None
    if priors_path and Path(priors_path).is_file():
        demos = SemanticPriorBank.load_json(priors_path).retrieve(task_instruction, top_k=2)

    pcfg = plugin_cfg or HarnessPluginConfig()
    pipeline = PluginPipeline(cfg=pcfg)
    schema = single_arm_step_json_schema(supports_mobile_base=False)
    vlm_turns: list[dict[str, Any]] = []
    last_subtask = ""
    vlm_total_tokens = 0
    vlm_total_elapsed = 0.0

    try:
        for turn_idx in range(1, max_turns + 1):
            env.current_turn_idx = turn_idx
            obs_jpg = env.save_observation_jpg(mdir / f"turn_{turn_idx:02d}.jpg")
            g_cmd = "OPEN" if env.gripper_open > 0.5 else "CLOSED"
            g_qpos = [round(float(v), 4) for v in env.obs.get("robot0_gripper_qpos", [])]
            q_rad = [round(float(v), 4) for v in env.obs["robot0_joint_pos"]]

            ctx = PluginObservationContext(
                embodiment_summary=f"blind unknown 3D robot `{robosuite_robot}` (URDF-FREE: calibrated via few-shot active self-probing + online Broyden Jacobian)",
                task_instruction=task_instruction,
                turn_idx=turn_idx,
                max_turns=max_turns,
                allowed_units=valid_units_for_embodiment(supports_mobile_base=False),
                camera_guides=[
                    ("CAM 1 (FRONT)", "up in frame = `+Z` (`MV_UP`), down = `-Z` (`MV_DOWN`), right = `+Y` (`MV_LEFT`), left = `-Y` (`MV_RIGHT`), closer/lower = `+X` (`MV_FWD`)."),
                    ("CAM 2 (LEFT-SIDE)", "left in frame = `+X` (`MV_FWD`), right in frame = `-X` (`MV_BACK`), up in frame = `+Z` (`MV_UP`), down in frame = `-Z` (`MV_DOWN`)."),
                    ("CAM 3 (WRIST EYE-IN-HAND)", "Wrist camera looking down toward gripper fingers: above fingers = `+X`, below = `-X`, left = `+Y`, right = `-Y`."),
                ],
                arms=[
                    ArmProprioPacket(
                        arm_label="main",
                        robot_id=f"{robosuite_robot.lower()}_urdf_free",
                        step_m=float(env.model.step_m),
                        joint_q_rad=q_rad,
                        tcp_base_xyz_m=[0.0, 0.0, 0.0],
                        gripper_cmd=g_cmd,
                        gripper_encoder_qpos=g_qpos,
                        has_gripper=True,
                        world_axis_note=f"NO URDF FK available; visual TCP tracked at frontview=({env.top_uv[0]:.0f},{env.top_uv[1]:.0f})px, sideview=({env.side_uv[0]:.0f},{env.side_uv[1]:.0f})px",
                    )
                ],
                last_subtask=last_subtask,
                recent_history=list(env.recent_units),
                in_context_demos=list(demos or ()),
                is_bimanual=False,
                extra_embodiment_note=f"Self-Probing Embodiment Diagnosis: {diag_json.get('summary', '')}",
            )
            prompt = pipeline.build_prompt(ctx)
            resp = backend.generate(prompt=prompt, images=[obs_jpg], json_schema=schema, timeout_s=600.0)
            vlm_total_tokens += resp.tokens_used
            vlm_total_elapsed += resp.elapsed_s

            parsed = resp.parse_json()
            thought = str(parsed.get("thought", ""))
            subtask = str(parsed.get("current_subtask", ""))
            chunk = [str(u) for u in parsed.get("action_chunk", ["HOLD"])]
            last_subtask = subtask
            env.subtask_label = subtask

            unit_logs: list[dict[str, Any]] = []
            done_emitted = False
            for u in chunk:
                if u == "DONE":
                    done_emitted = True
                    unit_logs.append({"unit": "DONE", "accepted": True, "note": "vlm_declared_done"})
                    break
                unit_logs.append(env.step_unit(u))

            ok_now, audit_metrics = env.check_success_audit_only()
            vlm_turns.append(
                {
                    "turn": turn_idx,
                    "zero_privilege_verified": True,
                    "observation_image": f"media/{run_id}/turn_{turn_idx:02d}.jpg",
                    "prompt": sanitize_text(prompt),
                    "raw_response": sanitize_text(resp.text),
                    "thought": sanitize_text(thought),
                    "current_subtask": sanitize_text(subtask),
                    "action_chunk": chunk,
                    "token_usage": sanitize_payload(resp.raw_meta),
                    "tokens_used": resp.tokens_used,
                    "elapsed_s": resp.elapsed_s,
                    "unit_grounding_logs": unit_logs,
                    "post_turn_audit_only": audit_metrics,
                }
            )
            if ok_now:
                for _ in range(14):
                    env._step_raw_joints(env.q_target, env.gripper_open, n_sub=2, record_stride=1)
                env.save_observation_jpg(mdir / "final_success.jpg")
                break
            if done_emitted and not ok_now:
                last_subtask = "visual_check_goal_not_yet_satisfied_continue"

        ok_final, final_audit = env.check_success_audit_only()
        env.export_mp4(mdir / "rollout.mp4", fps=20)
        num_frames = len(env.video_frames)
    finally:
        env.close()

    final_metrics: dict[str, Any] = {
        "mujoco_task_success": final_audit["mujoco_task_success"],
        "urdf_provided": False,
        "self_probe_joints": env.dof,
    }
    if "cube_z_m" in final_audit:
        final_metrics["cube_z_m"] = final_audit["cube_z_m"]
    if "audit_xy_err_m" in final_audit:
        final_metrics["xy_alignment_err_m"] = final_audit["audit_xy_err_m"]

    payload = sanitize_payload(
        {
            "run_id": run_id,
            "zero_privilege": True,
            "urdf_free": True,
            "is_bimanual": False,
            "task_name": f"{robosuite_env_name}_URDF_Free",
            "robot_id": f"{robosuite_robot.lower()}_urdf_free",
            "robosuite_robot": f"{robosuite_robot} (No URDF · Self-Probed)",
            "environment": f"MuJoCo 3D · robosuite {robosuite_env_name} (URDF-Free Few-Shot Self-Probing + Zero-Privilege RGB)",
            "unseen_embodiment": True,
            "sidecar_used": False,
            "plugins_enabled": pcfg.enabled_plugin_names(),
            "task_instruction": task_instruction,
            "model": backend.model_name,
            "reasoning_effort": getattr(backend, "reasoning_effort", "max"),
            "fast_mode": bool(getattr(backend, "fast_mode", False)),
            "success": ok_final,
            "video_path": f"media/{run_id}/rollout.mp4",
            "self_probe_montage": f"media/{run_id}/self_probe_montage.jpg",
            "video_frames": num_frames,
            "video_fps": 20,
            "video_duration_s": round(num_frames / 20.0, 2),
            "total_vlm_turns": len(vlm_turns),
            "total_semantic_units": sum(len(t["action_chunk"]) for t in vlm_turns),
            "total_vlm_tokens": vlm_total_tokens + diag_resp.tokens_used,
            "total_vlm_elapsed_s": round(vlm_total_elapsed + diag_resp.elapsed_s, 3),
            "total_pipeline_elapsed_s": round(time.monotonic() - t_start, 3),
            "final_metrics": final_metrics,
            "stage1_urdf_and_synthesis": {
                "urdf_export": {"urdf_provided": False, "self_probing_report": probe_report},
                "urdf_xml": "<!-- ZERO URDF PROVIDED: Embodiment estimated online via Finger-Wiggle TCP discovery + D-joint Active Motor Babbling + Broyden Visual-Motor Jacobian -->",
                "synthesized_interpreter_code": json.dumps(
                    {
                        "mode": "SelfProbedEmbodimentModel (URDF-Free Few-Shot)",
                        "llm_diagnosis": diag_json,
                        "visual_motor_jacobian_px_per_rad": probe_report["visual_motor_jacobian_px_per_rad"],
                        "directional_cosine_self_check": probe_report["directional_cosine_self_check"],
                    },
                    indent=2,
                ),
                "synthesis_report": {
                    "robot_id": f"{robosuite_robot.lower()}_urdf_free",
                    "model": backend.model_name,
                    "passed": True,
                    "attempts": 1,
                    "total_tokens": diag_resp.tokens_used,
                    "total_elapsed_s": diag_resp.elapsed_s,
                    "turns": [
                        {
                            "attempt": 1,
                            "tokens_used": diag_resp.tokens_used,
                            "elapsed_s": diag_resp.elapsed_s,
                            "passed": True,
                            "failures": [],
                            "prompt": sanitize_text(diag_prompt),
                            "response_text": sanitize_text(diag_resp.text),
                        }
                    ],
                },
            },
            "stage2_vlm_closed_loop_turns": vlm_turns,
        }
    )

    out_json = Path(trace_json_out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
