"""Core geometric, kinematic, capability, and diagnostic data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import math
import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix [v]_x."""
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    return np.array(
        [
            [0.0, -vz, vy],
            [vz, 0.0, -vx],
            [-vy, vx, 0.0],
        ],
        dtype=np.float64,
    )


def rpy_to_matrix(rpy: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw (extrinsic XYZ / intrinsic ZYX) to 3x3 rotation."""
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation formula for unit/non-unit axis and scalar angle (rad)."""
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3, dtype=np.float64)
    u = np.asarray(axis, dtype=np.float64) / norm
    K = skew(u)
    return np.eye(3, dtype=np.float64) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Convert 3D axis-angle vector (norm = angle) to 3x3 rotation matrix."""
    v = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    return axis_angle_to_matrix(v / angle, angle)


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Logarithm map on SO(3): returns 3D rotation vector in [-pi, pi]."""
    tr = float(np.trace(R))
    cos_theta = max(-1.0, min(1.0, 0.5 * (tr - 1.0)))
    theta = math.acos(cos_theta)
    if theta < 1e-8:
        return 0.5 * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            dtype=np.float64,
        )
    if math.pi - theta < 1e-5:
        B = 0.5 * (R + np.eye(3, dtype=np.float64))
        diag = np.maximum(0.0, np.diag(B))
        axis = np.sqrt(diag)
        k = int(np.argmax(axis))
        if axis[k] > 1e-12:
            # Recover consistent relative signs from symmetric column B[:, k] = u * u_k
            for j in range(3):
                if j != k and B[j, k] < 0.0:
                    axis[j] = -abs(axis[j])
                elif j != k:
                    axis[j] = abs(axis[j])
            # Disambiguate overall sign when slightly below pi via skew-symmetric part
            skew_vec = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
            if float(np.dot(axis, skew_vec)) < 0.0:
                axis = -axis
        n = float(np.linalg.norm(axis))
        if n > 1e-12:
            axis /= n
        return axis * theta
    factor = 0.5 * theta / math.sin(theta)
    return factor * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )


def quat_xyzw_to_matrix(q: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Convert [qx, qy, qz, qw] quaternion to 3x3 rotation matrix."""
    arr = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = arr / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_wxyz_to_matrix(q: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Convert [qw, qx, qy, qz] (MuJoCo convention) quaternion to 3x3 rotation matrix."""
    arr = np.asarray(q, dtype=np.float64)
    return quat_xyzw_to_matrix([arr[1], arr[2], arr[3], arr[0]])


def matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to unit quaternion [qx, qy, qz, qw] with qw >= 0."""
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[3] < 0.0:
        q = -q
    return q


@dataclass(frozen=True)
class Transform:
    """Rigid-body transformation in SE(3)."""

    R: np.ndarray  # (3, 3) float64
    p: np.ndarray  # (3,) float64

    @staticmethod
    def identity() -> Transform:
        return Transform(np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))

    @staticmethod
    def from_xyz_rpy(
        xyz: tuple[float, float, float] | list[float] | np.ndarray = (0.0, 0.0, 0.0),
        rpy: tuple[float, float, float] | list[float] | np.ndarray = (0.0, 0.0, 0.0),
    ) -> Transform:
        return Transform(
            R=rpy_to_matrix(rpy),
            p=np.asarray(xyz, dtype=np.float64).reshape(3),
        )

    @staticmethod
    def from_xyz_quat_xyzw(
        xyz: tuple[float, float, float] | list[float] | np.ndarray,
        quat_xyzw: tuple[float, float, float, float] | list[float] | np.ndarray,
    ) -> Transform:
        return Transform(
            R=quat_xyzw_to_matrix(quat_xyzw),
            p=np.asarray(xyz, dtype=np.float64).reshape(3),
        )

    @staticmethod
    def from_matrix(T: np.ndarray) -> Transform:
        arr = np.asarray(T, dtype=np.float64)
        return Transform(R=arr[:3, :3].copy(), p=arr[:3, 3].copy())

    def as_matrix(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3, 3] = self.p
        return T

    def compose(self, other: Transform) -> Transform:
        return Transform(R=self.R @ other.R, p=self.R @ other.p + self.p)

    def __matmul__(self, other: Transform) -> Transform:
        return self.compose(other)

    def inverse(self) -> Transform:
        Rt = self.R.T
        return Transform(R=Rt, p=-(Rt @ self.p))

    def transform_point(self, pt: np.ndarray | list[float]) -> np.ndarray:
        return self.R @ np.asarray(pt, dtype=np.float64).reshape(3) + self.p

    def transform_vector(self, vec: np.ndarray | list[float]) -> np.ndarray:
        return self.R @ np.asarray(vec, dtype=np.float64).reshape(3)

    @property
    def quat_xyzw(self) -> np.ndarray:
        return matrix_to_quat_xyzw(self.R)

    def pose7_xyzw(self) -> np.ndarray:
        return np.concatenate([self.p, self.quat_xyzw])

    def error_twist_world(self, target: Transform) -> np.ndarray:
        """6D spatial error [dp_world, drot_world] moving self toward target."""
        dp = target.p - self.p
        R_err = target.R @ self.R.T
        drot = matrix_to_rotvec(R_err)
        return np.concatenate([dp, drot])

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "xyz": [round(float(x), 6) for x in self.p],
            "quat_xyzw": [round(float(x), 6) for x in self.quat_xyzw],
        }


class EndEffectorKind(str, Enum):
    PARALLEL_GRIPPER = "parallel_gripper"
    SUCTION = "suction"
    DEXTEROUS_HAND = "dexterous_hand"
    NONE = "none"


@dataclass(frozen=True)
class EndEffectorSpec:
    kind: EndEffectorKind
    max_width: float = 0.08  # meters between open jaws
    finger_joints: tuple[str, ...] = ()
    closed_value: float = 0.0
    open_value: float = 1.0


@dataclass(frozen=True)
class MobileBaseSpec:
    mode: str = "holonomic_se2"  # holonomic_se2 | diff_drive
    joint_names: tuple[str, ...] = ("base_x", "base_y", "base_yaw")
    max_linear_vel: float = 0.5
    max_angular_vel: float = 1.0


@dataclass(frozen=True)
class ChainSpec:
    """One articulated manipulator chain on the robot body (e.g., 'main', 'left', 'right')."""

    name: str
    root_link: str
    tcp_link: str
    joint_names: tuple[str, ...]
    dof: int
    lower_limits: np.ndarray
    upper_limits: np.ndarray
    continuous_mask: np.ndarray
    home_q: np.ndarray
    end_effector: EndEffectorSpec
    reach_radius_m: float
    base_in_robot: Transform


@dataclass
class WorkspaceSummary:
    """Sampled reachability and bimanual overlap envelope computed directly from URDF FK."""

    voxel_size_m: float
    chain_bounds_min: dict[str, np.ndarray]
    chain_bounds_max: dict[str, np.ndarray]
    chain_voxel_counts: dict[str, int]
    bimanual_overlap_voxel_count: int = 0
    bimanual_overlap_centroid: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "voxel_size_m": self.voxel_size_m,
            "chain_bounds_min": {k: np.round(v, 4).tolist() for k, v in self.chain_bounds_min.items()},
            "chain_bounds_max": {k: np.round(v, 4).tolist() for k, v in self.chain_bounds_max.items()},
            "chain_voxel_counts": dict(self.chain_voxel_counts),
            "bimanual_overlap_voxel_count": self.bimanual_overlap_voxel_count,
            "bimanual_overlap_centroid": (
                np.round(self.bimanual_overlap_centroid, 4).tolist()
                if self.bimanual_overlap_centroid is not None
                else None
            ),
        }


@dataclass
class BodyCaps:
    """Structured morphology capability profile compiled from a URDF + optional sidecar."""

    robot_id: str
    chains: dict[str, ChainSpec]
    mobile_base: MobileBaseSpec | None = None
    torso_joints: tuple[str, ...] = ()
    torso_limits: tuple[np.ndarray, np.ndarray] | None = None
    control_hz: float = 20.0
    cameras: tuple[str, ...] = ()
    workspace: WorkspaceSummary | None = None

    @property
    def num_arms(self) -> int:
        return len(self.chains)

    @property
    def is_bimanual(self) -> bool:
        return len(self.chains) >= 2

    @property
    def is_mobile(self) -> bool:
        return self.mobile_base is not None

    @property
    def max_arm_dof(self) -> int:
        return max((c.dof for c in self.chains.values()), default=0)

    @property
    def min_arm_dof(self) -> int:
        return min((c.dof for c in self.chains.values()), default=0)

    @property
    def max_gripper_width(self) -> float:
        return max((c.end_effector.max_width for c in self.chains.values()), default=0.0)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "num_arms": self.num_arms,
            "is_bimanual": self.is_bimanual,
            "is_mobile": self.is_mobile,
            "torso_joints": list(self.torso_joints),
            "chains": {
                name: {
                    "dof": ch.dof,
                    "root_link": ch.root_link,
                    "tcp_link": ch.tcp_link,
                    "joints": list(ch.joint_names),
                    "continuous_joints": [
                        jn for jn, is_c in zip(ch.joint_names, ch.continuous_mask) if bool(is_c)
                    ],
                    "reach_radius_m": round(ch.reach_radius_m, 4),
                    "end_effector": {
                        "kind": ch.end_effector.kind.value,
                        "max_width": ch.end_effector.max_width,
                    },
                }
                for name, ch in self.chains.items()
            },
            "workspace": self.workspace.to_dict() if self.workspace is not None else None,
        }


class FailureCode(str, Enum):
    OUT_OF_REACH = "out_of_reach"
    IK_NO_CONVERGENCE = "ik_no_convergence"
    JOINT_LIMIT_VIOLATION = "joint_limit_violation"
    GRIPPER_WIDTH_EXCEEDED = "gripper_width_exceeded"
    CAPABILITY_MISMATCH = "capability_mismatch"
    BIMANUAL_NO_OVERLAP = "bimanual_no_overlap"
    SINGULARITY = "kinematic_singularity"


@dataclass
class LoweringDiagnostic:
    """Structured physical reason returned to the frontier model when lowering fails."""

    code: FailureCode
    message: str
    chain_name: str | None = None
    target_xyz: list[float] | None = None
    deficit_m: float | None = None
    position_error_m: float | None = None
    orientation_error_rad: float | None = None
    suggested_repair: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "chain_name": self.chain_name,
            "target_xyz": self.target_xyz,
            "deficit_m": round(self.deficit_m, 4) if self.deficit_m is not None else None,
            "position_error_m": (
                round(self.position_error_m, 5) if self.position_error_m is not None else None
            ),
            "orientation_error_rad": (
                round(self.orientation_error_rad, 5)
                if self.orientation_error_rad is not None
                else None
            ),
            "suggested_repair": self.suggested_repair,
            "details": self.details,
        }


class LoweringError(RuntimeError):
    """Raised when an EIR primitive cannot be lowered onto the target URDF body."""

    def __init__(self, diagnostic: LoweringDiagnostic) -> None:
        super().__init__(f"[{diagnostic.code.value}] {diagnostic.message}")
        self.diagnostic = diagnostic


@dataclass
class RobotWaypoint:
    """Compiled low-level setpoint for one control step across mobile base, torso, and arms."""

    arm_q: dict[str, np.ndarray]
    gripper: dict[str, float]  # normalized [0.0=closed, 1.0=open]
    base_se2: np.ndarray | None = None  # [x, y, yaw] in world frame
    torso_q: np.ndarray | None = None
    label: str = ""


@dataclass
class PrimitiveTrace:
    """Per-primitive execution & lowering trace exposed to the coding agent."""

    skill_name: str
    chain_used: str
    waypoints: list[RobotWaypoint]
    achieved_tcp: dict[str, Transform]
    ik_pos_err_m: float
    ik_rot_err_rad: float
    notes: list[str] = field(default_factory=list)
