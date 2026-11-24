"""Standard URDF and MuJoCo MJCF kinematic tree parser."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np

from eaisys.types import Transform, quat_wxyz_to_matrix


def _parse_floats(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if not text or not text.strip():
        return np.array(default, dtype=np.float64)
    return np.array([float(x) for x in text.strip().split()], dtype=np.float64)


@dataclass(frozen=True)
class LinkNode:
    name: str
    mass: float = 0.0


@dataclass(frozen=True)
class JointNode:
    name: str
    joint_type: str  # revolute | continuous | prismatic | fixed
    parent: str
    child: str
    origin: Transform
    axis: np.ndarray  # unit (3,) in joint frame
    lower: float = -math.pi
    upper: float = math.pi
    velocity: float = 2.0
    effort: float = 100.0
    mimic_joint: str | None = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0

    @property
    def is_actuated(self) -> bool:
        return self.joint_type in ("revolute", "continuous", "prismatic") and self.mimic_joint is None

    @property
    def is_continuous(self) -> bool:
        return self.joint_type == "continuous"


@dataclass
class KinematicTree:
    name: str
    links: dict[str, LinkNode]
    joints: dict[str, JointNode]
    source_path: str = ""
    children_map: dict[str, list[JointNode]] = field(init=False)
    parent_joint_map: dict[str, JointNode] = field(init=False)
    root_link: str = field(init=False)

    def __post_init__(self) -> None:
        self.children_map = {lk: [] for lk in self.links}
        self.parent_joint_map = {}
        child_links = set()
        for jt in self.joints.values():
            self.children_map.setdefault(jt.parent, []).append(jt)
            self.parent_joint_map[jt.child] = jt
            child_links.add(jt.child)
        roots = [lk for lk in self.links if lk not in child_links]
        if not roots:
            raise ValueError(f"KinematicTree '{self.name}' has no root link (cycle detected).")
        self.root_link = roots[0]

    def joint_path(self, start_link: str, end_link: str) -> list[JointNode]:
        """Return ordered joints along the unique directed path from start_link to end_link."""
        if start_link == end_link:
            return []
        path: list[JointNode] = []
        cur = end_link
        visited = set()
        while cur != start_link:
            if cur in visited or cur not in self.parent_joint_map:
                raise ValueError(
                    f"Link '{end_link}' is not a descendant of '{start_link}' in '{self.name}'."
                )
            visited.add(cur)
            jt = self.parent_joint_map[cur]
            path.append(jt)
            cur = jt.parent
        path.reverse()
        return path

    def leaf_links(self) -> list[str]:
        return [lk for lk, ch in self.children_map.items() if len(ch) == 0]


def parse_urdf_xml(root: ET.Element, source_path: str = "") -> KinematicTree:
    robot_name = root.get("name", "unnamed_robot")
    links: dict[str, LinkNode] = {}
    joints: dict[str, JointNode] = {}

    for lk_el in root.findall("link"):
        name = lk_el.get("name")
        if not name:
            continue
        mass = 0.0
        inertial = lk_el.find("inertial/mass")
        if inertial is not None and inertial.get("value") is not None:
            mass = float(inertial.get("value", "0.0"))
        links[name] = LinkNode(name=name, mass=mass)

    for jt_el in root.findall("joint"):
        name = jt_el.get("name")
        jtype = jt_el.get("type", "fixed").strip().lower()
        parent_el = jt_el.find("parent")
        child_el = jt_el.find("child")
        if not name or parent_el is None or child_el is None:
            continue
        parent = parent_el.get("link", "")
        child = child_el.get("link", "")
        links.setdefault(parent, LinkNode(name=parent))
        links.setdefault(child, LinkNode(name=child))

        orig_el = jt_el.find("origin")
        xyz = _parse_floats(orig_el.get("xyz") if orig_el is not None else None, (0.0, 0.0, 0.0))
        rpy = _parse_floats(orig_el.get("rpy") if orig_el is not None else None, (0.0, 0.0, 0.0))
        origin = Transform.from_xyz_rpy(xyz, rpy)

        axis_el = jt_el.find("axis")
        axis = _parse_floats(axis_el.get("xyz") if axis_el is not None else None, (1.0, 0.0, 0.0))
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm > 1e-12:
            axis = axis / axis_norm
        else:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        lim_el = jt_el.find("limit")
        if jtype == "continuous":
            lower, upper = -math.pi, math.pi
        else:
            lower = float(lim_el.get("lower", "-3.141592653589793")) if lim_el is not None else -math.pi
            upper = float(lim_el.get("upper", "3.141592653589793")) if lim_el is not None else math.pi
        vel = float(lim_el.get("velocity", "2.0")) if lim_el is not None else 2.0
        eff = float(lim_el.get("effort", "100.0")) if lim_el is not None else 100.0

        mimic_el = jt_el.find("mimic")
        mimic_joint = mimic_el.get("joint") if mimic_el is not None else None
        mimic_mult = float(mimic_el.get("multiplier", "1.0")) if mimic_el is not None else 1.0
        mimic_off = float(mimic_el.get("offset", "0.0")) if mimic_el is not None else 0.0

        joints[name] = JointNode(
            name=name,
            joint_type=jtype,
            parent=parent,
            child=child,
            origin=origin,
            axis=axis,
            lower=lower,
            upper=upper,
            velocity=vel,
            effort=eff,
            mimic_joint=mimic_joint,
            mimic_multiplier=mimic_mult,
            mimic_offset=mimic_off,
        )

    return KinematicTree(name=robot_name, links=links, joints=joints, source_path=source_path)


def parse_mjcf_xml(root: ET.Element, source_path: str = "") -> KinematicTree:
    """Import a MuJoCo MJCF robot.xml (e.g., robosuite/LIBERO/RoboCasa) into KinematicTree."""
    robot_name = root.get("model", "mjcf_robot")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF '{source_path}' has no <worldbody>.")

    compiler_el = root.find("compiler")
    angle_mode = (
        compiler_el.get("angle", "degree").strip().lower()
        if compiler_el is not None
        else "degree"
    )
    ang_scale = 1.0 if angle_mode.startswith("rad") else (math.pi / 180.0)

    links: dict[str, LinkNode] = {"world": LinkNode(name="world")}
    joints: dict[str, JointNode] = {}

    def _body_transform(el: ET.Element) -> Transform:
        pos = _parse_floats(el.get("pos"), (0.0, 0.0, 0.0))
        if el.get("quat") is not None:
            q_wxyz = _parse_floats(el.get("quat"), (1.0, 0.0, 0.0, 0.0))
            R = quat_wxyz_to_matrix(q_wxyz)
            return Transform(R=R, p=pos)
        if el.get("euler") is not None:
            rpy = _parse_floats(el.get("euler"), (0.0, 0.0, 0.0)) * ang_scale
            return Transform.from_xyz_rpy(pos, rpy)
        return Transform.from_xyz_rpy(pos, (0.0, 0.0, 0.0))

    counter = 0

    def _walk(parent_link: str, container: ET.Element) -> None:
        nonlocal counter
        for b_el in container.findall("body"):
            counter += 1
            b_name = b_el.get("name") or f"body_{counter}"
            links[b_name] = LinkNode(name=b_name)
            T_body = _body_transform(b_el)
            b_joints = b_el.findall("joint")
            if not b_joints:
                jt_name = f"fixed_{parent_link}_to_{b_name}"
                joints[jt_name] = JointNode(
                    name=jt_name,
                    joint_type="fixed",
                    parent=parent_link,
                    child=b_name,
                    origin=T_body,
                    axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                )
            else:
                entry_link = f"{b_name}__frame_in"
                links[entry_link] = LinkNode(name=entry_link)
                joints[f"fixed_{parent_link}_to_{entry_link}"] = JointNode(
                    name=f"fixed_{parent_link}_to_{entry_link}",
                    joint_type="fixed",
                    parent=parent_link,
                    child=entry_link,
                    origin=T_body,
                    axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                )
                prev_link = entry_link
                for idx, j_el in enumerate(b_joints):
                    j_name = j_el.get("name") or f"{b_name}_joint_{idx}"
                    mj_type = j_el.get("type", "hinge").strip().lower()
                    axis = _parse_floats(j_el.get("axis"), (0.0, 0.0, 1.0))
                    n_ax = float(np.linalg.norm(axis))
                    axis = axis / n_ax if n_ax > 1e-12 else np.array([0.0, 0.0, 1.0])
                    j_pos = _parse_floats(j_el.get("pos"), (0.0, 0.0, 0.0))

                    rng_str = j_el.get("range")
                    if mj_type == "slide":
                        jtype = "prismatic"
                        lo, hi = (
                            (-0.5, 0.5)
                            if rng_str is None
                            else tuple(_parse_floats(rng_str, (-0.5, 0.5))[:2])
                        )
                    elif rng_str is None:
                        jtype = "continuous"
                        lo, hi = -math.pi, math.pi
                    else:
                        jtype = "revolute"
                        raw_lo, raw_hi = tuple(_parse_floats(rng_str, (-180.0, 180.0))[:2])
                        lo, hi = raw_lo * ang_scale, raw_hi * ang_scale

                    rot_link = f"{b_name}__piv_{idx}"
                    links[rot_link] = LinkNode(name=rot_link)
                    joints[j_name] = JointNode(
                        name=j_name,
                        joint_type=jtype,
                        parent=prev_link,
                        child=rot_link,
                        origin=Transform.from_xyz_rpy(j_pos),
                        axis=axis,
                        lower=float(lo),
                        upper=float(hi),
                    )
                    after_link = b_name if idx == len(b_joints) - 1 else f"{b_name}__inter_{idx}"
                    links.setdefault(after_link, LinkNode(name=after_link))
                    joints[f"fixed_{rot_link}_unpivot"] = JointNode(
                        name=f"fixed_{rot_link}_unpivot",
                        joint_type="fixed",
                        parent=rot_link,
                        child=after_link,
                        origin=Transform.from_xyz_rpy(-j_pos),
                        axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                    )
                    prev_link = after_link

            for s_el in b_el.findall("site"):
                s_name = s_el.get("name")
                if not s_name:
                    continue
                site_link = f"site::{s_name}"
                links[site_link] = LinkNode(name=site_link)
                joints[f"fixed_to_{site_link}"] = JointNode(
                    name=f"fixed_to_{site_link}",
                    joint_type="fixed",
                    parent=b_name,
                    child=site_link,
                    origin=_body_transform(s_el),
                    axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                )

            _walk(b_name, b_el)

    _walk("world", worldbody)
    return KinematicTree(name=robot_name, links=links, joints=joints, source_path=source_path)


def parse_robot_file(path: str | Path) -> KinematicTree:
    """Parse either a standard URDF (.urdf / .xml) or a MuJoCo MJCF (<mujoco>) file."""
    p = Path(path)
    tree = ET.parse(p)
    root = tree.getroot()
    tag = root.tag.strip().lower()
    if tag == "robot":
        return parse_urdf_xml(root, source_path=str(p))
    if tag == "mujoco":
        return parse_mjcf_xml(root, source_path=str(p))
    raise ValueError(f"Unsupported robot description root tag '<{root.tag}>' in {p}")
