"""Unified trajectory episode schema across LeRobot v2/v3, RoboMIND HDF5, ALOHA HDF5, and AgiBot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class ArmTrajectoryStream:
    """Time-aligned proprioceptive stream for one manipulator arm over T frames."""

    chain_name: str
    qpos: np.ndarray | None = None  # (T, D) radians
    eef_pose7_xyzw: np.ndarray | None = None  # (T, 7) [x, y, z, qx, qy, qz, qw]
    gripper_open: np.ndarray | None = None  # (T,) normalized in [0.0=closed, 1.0=open]

    @property
    def num_frames(self) -> int:
        for arr in (self.qpos, self.eef_pose7_xyzw, self.gripper_open):
            if arr is not None:
                return int(len(arr))
        return 0


@dataclass
class GroundTruthSkillSpan:
    start_frame: int
    end_frame: int
    skill: str
    action_text: str


@dataclass
class RawEpisode:
    """Normalized multi-embodiment episode ready for URDF kinematic handshake and EIR lifting."""

    dataset_name: str
    embodiment_id: str
    episode_id: str
    task_instruction: str
    fps: float
    arms: dict[str, ArmTrajectoryStream]
    base_pose7_xyzw: np.ndarray | None = None  # (T, 7) if mobile
    base_twist6: np.ndarray | None = None  # (T, 6) if mobile
    gt_skill_spans: list[GroundTruthSkillSpan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        if not self.arms:
            return len(self.gt_skill_spans)
        return max(s.num_frames for s in self.arms.values())


def normalize_signal_01(values: np.ndarray, min_span: float = 1e-3) -> np.ndarray:
    """Normalize arbitrary continuous gripper telemetry into [0.0=closed, 1.0=open]."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    lo, hi = float(np.percentile(arr, 2)), float(np.percentile(arr, 98))
    if hi - lo < min_span:
        # Constant gripper throughout episode -> treat as open if positive else closed
        return np.ones_like(arr) if float(np.median(arr)) >= 0.0 else np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
