"""Minimal physical URDF substrate: XML/MJCF parser, numerical FK/Jacobian/IK, and handshake."""

from pathlib import Path

from eaisys.urdf.handshake import HandshakeReport, kinematic_handshake
from eaisys.urdf.kinematics import IKResult, KinematicChainSolver, forward_kinematics_tree
from eaisys.urdf.parser import JointNode, KinematicTree, LinkNode, parse_robot_file


def default_robots_root() -> Path:
    pkg_bundled = Path(__file__).resolve().parents[1] / "_bundled" / "robots"
    if pkg_bundled.is_dir():
        return pkg_bundled
    return Path(__file__).resolve().parents[3] / "robots"


def default_harnesses_root() -> Path:
    pkg_bundled = Path(__file__).resolve().parents[1] / "_bundled" / "harnesses"
    if pkg_bundled.is_dir():
        return pkg_bundled
    return Path(__file__).resolve().parents[3] / "harnesses"


__all__ = [
    "HandshakeReport",
    "IKResult",
    "JointNode",
    "KinematicChainSolver",
    "KinematicTree",
    "LinkNode",
    "default_harnesses_root",
    "default_robots_root",
    "forward_kinematics_tree",
    "kinematic_handshake",
    "parse_robot_file",
]
