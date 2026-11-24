"""Zero-Privilege 3D MuJoCo (robosuite) single-arm & heterogeneous bimanual closed-loop VLM execution.

STRICT ZERO-PRIVILEGE GUARANTEE:
- The VLM receives ONLY:
  (1) Raw multi-camera RGB images with ZERO metric/height/holding overlays, and
  (2) Pure robot hardware encoder readouts (`joint_pos` in rad, raw `gripper_qpos`, and `fk_tcp(q)` in each arm's `base_link` frame).
- ZERO scene/environment privileges: NO object coordinates, NO table height, NO object dimensions,
  NO `HOLDING` classification thresholds, and NO table-floor workspace clamps.
- Every VLM prompt is built through the unified ablation-grade `PluginPipeline` and checked by
  `assert_zero_privilege_prompt()` before dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw

from eaisys.backends.base import FrontierBackend
from eaisys.harness.action_space import (
    bimanual_step_json_schema,
    single_arm_step_json_schema,
    valid_units_for_embodiment,
)
from eaisys.harness.interpreter import (
    EmbodimentInterpreter,
    load_interpreter_module,
    verify_interpreter_isolated,
)
from eaisys.harness.plugins import (
    ArmProprioPacket,
    HarnessPluginConfig,
    PluginObservationContext,
    PluginPipeline,
)
from eaisys.sanitize import sanitize_payload, sanitize_text
from eaisys.synthesis import SemanticPriorBank, URDFHarnessSynthesizer
from eaisys.types import quat_wxyz_to_matrix
from eaisys.urdf import forward_kinematics_tree, parse_robot_file


def _matrix_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    sp = float(-R[2, 0])
    sp = max(-1.0, min(1.0, sp))
    if abs(sp) < 0.999999:
        pitch = math.asin(sp)
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        pitch = math.copysign(math.pi / 2.0, sp)
        roll = math.atan2(float(-R[1, 2]), float(R[1, 1]))
        yaw = 0.0
    return (roll, pitch, yaw)


def _fmt_floats(vals: list[float] | tuple[float, ...] | np.ndarray) -> str:
    return " ".join(f"{float(v):.7g}" for v in vals)


def export_robosuite_urdf(
    robosuite_robot: str,
    robot_id: str,
    out_urdf_path: str | Path,
    *,
    env_name: str = "Lift",
) -> dict[str, Any]:
    """Export an unseen robosuite robot's arm + end-effector kinematic chain to standard `.urdf`."""
    import robosuite as suite
    from robosuite.controllers import load_part_controller_config
    from robosuite.controllers.composite.composite_controller_factory import (
        refactor_composite_controller_config,
    )

    arm_cfg = load_part_controller_config(default_controller="JOINT_POSITION")
    ctrl_cfg = refactor_composite_controller_config(arm_cfg, robosuite_robot, ["right"])
    export_env = "Wipe" if env_name == "Wipe" else "Lift"
    env = suite.make(
        env_name=export_env,
        robots=robosuite_robot,
        controller_configs=ctrl_cfg,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    obs = env.reset()
    m = env.sim.model
    d = env.sim.data
    robot = env.robots[0]

    def clean_name(raw: str) -> str:
        s = raw
        for pref in ("robot0_", "gripper0_right_", "gripper0_"):
            if s.startswith(pref):
                s = s[len(pref) :]
        return s

    root_el = ET.Element("robot", {"name": robot_id})
    added_links: set[str] = set()

    def ensure_link(lname: str) -> None:
        if lname not in added_links:
            ET.SubElement(root_el, "link", {"name": lname})
            added_links.add(lname)

    first_jid = m.joint_name2id(robot.robot_joints[0])
    first_parent_bid = m.body_parentid[m.jnt_bodyid[first_jid]]
    base_link_name = clean_name(m.body_id2name(first_parent_bid))
    ensure_link(base_link_name)
    base_world_pos = d.get_body_xpos(m.body_id2name(first_parent_bid)).copy()

    clean_joint_names: list[str] = []
    for jname in robot.robot_joints:
        jid = m.joint_name2id(jname)
        bid = m.jnt_bodyid[jid]
        pbid = m.body_parentid[bid]
        p_link = clean_name(m.body_id2name(pbid))
        c_link = clean_name(m.body_id2name(bid))
        ensure_link(p_link)
        ensure_link(c_link)

        b_pos = m.body_pos[bid] + quat_wxyz_to_matrix(m.body_quat[bid]) @ m.jnt_pos[jid]
        b_rpy = _matrix_to_rpy(quat_wxyz_to_matrix(m.body_quat[bid]))
        j_axis = m.jnt_axis[jid]
        is_lim = bool(m.jnt_limited[jid])
        jtype = "revolute" if is_lim else "continuous"
        cj_name = clean_name(jname)
        clean_joint_names.append(cj_name)

        j_el = ET.SubElement(root_el, "joint", {"name": cj_name, "type": jtype})
        ET.SubElement(j_el, "origin", {"xyz": _fmt_floats(b_pos), "rpy": _fmt_floats(b_rpy)})
        ET.SubElement(j_el, "parent", {"link": p_link})
        ET.SubElement(j_el, "child", {"link": c_link})
        ET.SubElement(j_el, "axis", {"xyz": _fmt_floats(j_axis)})
        if is_lim:
            lo, hi = float(m.jnt_range[jid, 0]), float(m.jnt_range[jid, 1])
            ET.SubElement(
                j_el,
                "limit",
                {"lower": f"{lo:.6f}", "upper": f"{hi:.6f}", "effort": "100", "velocity": "2.5"},
            )

    last_arm_bid = m.jnt_bodyid[m.joint_name2id(robot.robot_joints[-1])]
    sid = m.site_name2id("gripper0_right_grip_site")
    site_bid = m.site_bodyid[sid]

    body_chain: list[int] = []
    cur = site_bid
    while cur != last_arm_bid and cur != 0:
        body_chain.append(cur)
        cur = m.body_parentid[cur]
    body_chain.reverse()

    prev_link = clean_name(m.body_id2name(last_arm_bid))
    for bid in body_chain:
        cname = clean_name(m.body_id2name(bid))
        ensure_link(cname)
        b_pos = m.body_pos[bid]
        b_rpy = _matrix_to_rpy(quat_wxyz_to_matrix(m.body_quat[bid]))
        fj = ET.SubElement(root_el, "joint", {"name": f"fixed_{prev_link}_to_{cname}", "type": "fixed"})
        ET.SubElement(fj, "origin", {"xyz": _fmt_floats(b_pos), "rpy": _fmt_floats(b_rpy)})
        ET.SubElement(fj, "parent", {"link": prev_link})
        ET.SubElement(fj, "child", {"link": cname})
        prev_link = cname

    tcp_link = f"{robot_id}_tcp"
    ensure_link(tcp_link)
    s_pos = m.site_pos[sid]
    s_rpy = _matrix_to_rpy(quat_wxyz_to_matrix(m.site_quat[sid]))
    tcp_j = ET.SubElement(root_el, "joint", {"name": "fixed_to_tcp", "type": "fixed"})
    ET.SubElement(tcp_j, "origin", {"xyz": _fmt_floats(s_pos), "rpy": _fmt_floats(s_rpy)})
    ET.SubElement(tcp_j, "parent", {"link": prev_link})
    ET.SubElement(tcp_j, "child", {"link": tcp_link})

    finger_upper = 0.0425
    grip_obj = robot.gripper.get("right") if isinstance(robot.gripper, dict) else robot.gripper
    if grip_obj is not None and getattr(grip_obj, "joints", None):
        g_jids = [m.joint_name2id(jn) for jn in grip_obj.joints if jn in m._joint_name2id]
        if g_jids and all(int(m.jnt_type[jid]) == 2 for jid in g_jids):
            spans = [
                abs(float(m.jnt_range[jid, 1] - m.jnt_range[jid, 0]))
                for jid in g_jids
                if bool(m.jnt_limited[jid])
            ]
            if spans and max(spans) > 0.005:
                finger_upper = round(max(spans), 6)

    if export_env != "Wipe":
        for side, sign in (("left", 1), ("right", -1)):
            flk = f"{side}_finger_link"
            ensure_link(flk)
            fgj = ET.SubElement(root_el, "joint", {"name": f"{side}_finger_joint", "type": "prismatic"})
            ET.SubElement(fgj, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            ET.SubElement(fgj, "parent", {"link": tcp_link})
            ET.SubElement(fgj, "child", {"link": flk})
            ET.SubElement(fgj, "axis", {"xyz": f"0 {sign} 0"})
            ET.SubElement(
                fgj,
                "limit",
                {"lower": "0.0", "upper": f"{finger_upper:.6g}", "effort": "20", "velocity": "0.2"},
            )
            if side == "right":
                ET.SubElement(
                    fgj, "mimic", {"joint": "left_finger_joint", "multiplier": "1.0", "offset": "0.0"}
                )

    ET.indent(root_el, space="  ")
    xml_text = '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root_el, encoding="unicode") + "\n"
    out_p = Path(out_urdf_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(xml_text, encoding="utf-8")

    tree = parse_robot_file(out_p)
    q_init = obs["robot0_joint_pos"].copy()
    jmap = {jn: float(qv) for jn, qv in zip(clean_joint_names, q_init)}
    urdf_tcp_in_base = forward_kinematics_tree(tree, jmap)[tcp_link].p
    mj_tcp_in_base = d.get_site_xpos("gripper0_right_grip_site") - base_world_pos
    fk_err_m = float(np.linalg.norm(urdf_tcp_in_base - mj_tcp_in_base))
    env.close()
    if fk_err_m > 5e-4:
        raise RuntimeError(f"Exported URDF FK mismatch vs MuJoCo: err={fk_err_m:.6f}m")

    return {
        "robosuite_robot": robosuite_robot,
        "robot_id": robot_id,
        "source_env": export_env,
        "urdf_path": out_p.name,
        "actuated_joints": clean_joint_names,
        "tcp_link": tcp_link,
        "mujoco_fk_agreement_err_m": round(fk_err_m, 7),
    }


_CAMERA_PANEL_TITLE: dict[str, str] = {
    "agentview": "FRONT-TOP",
    "frontview": "FRONT",
    "birdview": "OVERHEAD TOP-DOWN",
    "sideview": "LEFT-SIDE",
    "robot0_robotview": "ROBOT-FRONT",
    "robot0_eye_in_hand": "WRIST EYE-IN-HAND",
    "robot1_eye_in_hand": "LEFT ARM WRIST",
}

_CAMERA_AXIS_GUIDE: dict[str, str] = {
    "agentview": "Front-top angled view looking toward robot base: lower in frame = `+X` (`MV_FWD`), higher in frame = `-X` (`MV_BACK`), right in frame = `+Y` (`MV_LEFT`), left in frame = `-Y` (`MV_RIGHT`).",
    "frontview": "Frontal view looking toward robot base: up in frame = `+Z` (`MV_UP`), down in frame = `-Z` (`MV_DOWN`), right in frame = `+Y` (`MV_LEFT`), left in frame = `-Y` (`MV_RIGHT`), closer/lower foreground = `+X` (`MV_FWD`).",
    "birdview": "Overhead top-down view: down in frame = `+X` (`MV_FWD`, away from robot base), up in frame = `-X` (`MV_BACK`, toward robot base), right in frame = `+Y` (`MV_LEFT`), left in frame = `-Y` (`MV_RIGHT`).",
    "sideview": "Left-side elevation view: left in frame = `+X` (`MV_FWD`), right in frame = `-X` (`MV_BACK`), up in frame = `+Z` (`MV_UP`), down in frame = `-Z` (`MV_DOWN`).",
    "robot0_robotview": "Front-angled view toward robot: lower in frame = `+X` (`MV_FWD`), higher = `-X` (`MV_BACK`), right = `+Y` (`MV_LEFT`), left = `-Y` (`MV_RIGHT`).",
    "robot0_eye_in_hand": "Wrist camera looking down toward end-effector: in default wrist yaw, above fingers/pad = `+X` (`MV_FWD`), below = `-X` (`MV_BACK`), left of center = `+Y` (`MV_LEFT`), right of center = `-Y` (`MV_RIGHT`).",
    "robot1_eye_in_hand": "Left-arm wrist camera looking down toward its gripper fingers: above fingers = `+X` (`MV_FWD`), below = `-X` (`MV_BACK`), left of center = `+Y` (`MV_LEFT`), right of center = `-Y` (`MV_RIGHT`).",
}


def default_cameras_for_env(robosuite_env_name: str) -> tuple[str, str, str]:
    """Select 3 standard fixed cell/wrist cameras supported by the arena (zero scene state)."""
    if robosuite_env_name == "Door":
        return ("frontview", "sideview", "robot0_eye_in_hand")
    if robosuite_env_name.startswith("PickPlace"):
        return ("frontview", "birdview", "robot0_eye_in_hand")
    if robosuite_env_name.startswith("NutAssembly"):
        return ("birdview", "sideview", "robot0_eye_in_hand")
    if robosuite_env_name == "Wipe":
        return ("birdview", "sideview", "robot0_eye_in_hand")
    return ("agentview", "sideview", "robot0_eye_in_hand")


def _export_urdf_subprocess(robosuite_robot: str, robot_id: str, urdf_path: Path, env_name: str = "Lift") -> dict[str, Any]:
    """Export robosuite URDF in an isolated subprocess so macOS CGL state stays clean."""
    import subprocess
    import sys as _sys

    sub_code = (
        "import json, sys; from eaisys.harness.mujoco_env import export_robosuite_urdf; "
        "print('__EXPORT_JSON__:' + json.dumps(export_robosuite_urdf(sys.argv[1], sys.argv[2], sys.argv[3], env_name=sys.argv[4])))"
    )
    proc = subprocess.run(
        [_sys.executable, "-c", sub_code, robosuite_robot, robot_id, str(urdf_path), env_name],
        capture_output=True,
        text=True,
        check=True,
    )
    marker_line = [ln for ln in proc.stdout.splitlines() if ln.startswith("__EXPORT_JSON__:")]
    return json.loads(marker_line[-1].split("__EXPORT_JSON__:", 1)[1])


@dataclass
class MuJoCoPhysicalHarnessEnv:
    """Zero-privilege 3D MuJoCo single-arm environment: ONLY 3 RGB cameras + robot encoder proprioception."""

    robosuite_env_name: str
    robosuite_robot: str
    interpreter: EmbodimentInterpreter
    task_instruction: str
    camera_size: int = 384
    substeps_per_unit: int = 14
    seed: int = 7
    camera_names: tuple[str, str, str] | None = None
    env_kwargs: dict[str, Any] = field(default_factory=dict)

    _env: Any = field(init=False, repr=False)
    active_cameras: tuple[str, str, str] = field(init=False)
    obs: dict[str, Any] = field(init=False, repr=False)
    state: Any = field(init=False)
    video_frames: list[np.ndarray] = field(default_factory=list, repr=False)
    step_count: int = 0
    recent_units: list[str] = field(default_factory=list)
    current_turn_idx: int = 0
    current_subtask_label: str = "initializing"
    last_unit_label: str = "RESET"

    def __post_init__(self) -> None:
        import robosuite as suite
        from robosuite.controllers import load_part_controller_config
        from robosuite.controllers.composite.composite_controller_factory import (
            refactor_composite_controller_config,
        )

        self.active_cameras = self.camera_names or default_cameras_for_env(self.robosuite_env_name)
        arm_cfg = load_part_controller_config(default_controller="JOINT_POSITION")
        arm_cfg["kp"] = 180
        ctrl_cfg = refactor_composite_controller_config(arm_cfg, self.robosuite_robot, ["right"])
        self._env = suite.make(
            env_name=self.robosuite_env_name,
            robots=self.robosuite_robot,
            controller_configs=ctrl_cfg,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=list(self.active_cameras),
            camera_heights=self.camera_size,
            camera_widths=self.camera_size,
            control_freq=20,
            horizon=2500,
            **self.env_kwargs,
        )
        self.reset()

    def _urdf_kinematic_reach_m(self) -> float:
        tree = getattr(self.interpreter, "tree", None)
        if tree is not None:
            total = sum(float(np.linalg.norm(jt.origin.p)) for jt in tree.joints.values())
            if total > 0.2:
                return round(total + 0.05, 3)
        return 1.25

    def reset(self) -> None:
        np.random.seed(self.seed)
        self.obs = self._env.reset()

        self.state = self.interpreter.reset()
        arm = self.interpreter.arm_names[0]
        self.state.arm_q[arm] = self.obs["robot0_joint_pos"].copy()
        self.state.tcp_setpoint[arm] = self.interpreter.fk_tcp(self.state, arm=arm)
        self.state.gripper_open[arm] = 1.0
        self.state.step_m = 0.02

        reach = self._urdf_kinematic_reach_m()
        self.interpreter.workspace_bounds = (
            np.array([-reach, -reach, -reach], dtype=np.float64),
            np.array([reach, reach, reach], dtype=np.float64),
        )

        self.video_frames.clear()
        self.step_count = 0
        self.recent_units.clear()
        self.current_turn_idx = 0
        self.current_subtask_label = "ready"
        self.last_unit_label = "RESET"
        grip_speed = float(getattr(self._env.robots[0].gripper.get("right"), "speed", 0.1)) if hasattr(self._env.robots[0], "gripper") else 0.1
        n_init = 120 if grip_speed < 0.03 else 6
        rec_stride = max(1, n_init // 6)
        for i in range(n_init):
            self._step_physics(self.state.arm_q[arm], gripper_open=1.0, record=(i % rec_stride == 0 or i == n_init - 1))

    def _joint_error(self, q_target: np.ndarray, q_cur: np.ndarray) -> np.ndarray:
        diff = np.asarray(q_target, dtype=np.float64) - np.asarray(q_cur, dtype=np.float64)
        robot = self._env.robots[0]
        m = self._env.sim.model
        for i, jname in enumerate(robot.robot_joints):
            jid = m.joint_name2id(jname)
            if not bool(m.jnt_limited[jid]):
                diff[i] = (diff[i] + math.pi) % (2.0 * math.pi) - math.pi
        return diff

    def _step_physics(self, q_target: np.ndarray, gripper_open: float, record: bool = True) -> None:
        q_cur = self.obs["robot0_joint_pos"]
        act_arm = np.clip(self._joint_error(q_target, q_cur) / 0.05, -1.0, 1.0)
        if self._env.action_dim == len(act_arm):
            act = act_arm
        else:
            g_cmd = -1.0 if gripper_open > 0.5 else 1.0
            act = np.concatenate([act_arm, [g_cmd]])
        self.obs, _, _, _ = self._env.step(act)
        if record:
            self.video_frames.append(self.compose_frame_rgb())

    def proprio_tcp_in_base(self) -> np.ndarray:
        arm = self.interpreter.arm_names[0]
        self.state.arm_q[arm] = self.obs["robot0_joint_pos"].copy()
        return self.interpreter.fk_tcp(self.state, arm=arm).p.copy()

    def raw_gripper_proprio(self) -> tuple[str, list[float]]:
        arm = self.interpreter.arm_names[0]
        if "robot0_gripper_qpos" not in self.obs or self._env.action_dim == len(self.obs["robot0_joint_pos"]):
            return "FIXED_TOOL", []
        cmd_str = "OPEN" if float(self.state.gripper_open.get(arm, 1.0)) > 0.5 else "CLOSED"
        raw_q = [round(float(v), 4) for v in self.obs["robot0_gripper_qpos"]]
        return cmd_str, raw_q

    def build_plugin_context(
        self,
        *,
        turn_idx: int,
        max_turns: int,
        last_subtask: str,
        in_context_demos: list[dict[str, Any]] | None,
        has_prev_visual_frame: bool = False,
    ) -> PluginObservationContext:
        interp = self.interpreter
        units = valid_units_for_embodiment(supports_mobile_base=interp.supports_mobile_base)
        tcp_b = self.proprio_tcp_in_base()
        g_cmd, g_qpos = self.raw_gripper_proprio()
        q_rad = [round(float(v), 4) for v in self.obs["robot0_joint_pos"]]
        has_grip = g_cmd != "FIXED_TOOL"
        cam_guides = [
            (
                f"CAM {idx} ({_CAMERA_PANEL_TITLE.get(c, c.upper())})",
                _CAMERA_AXIS_GUIDE.get(c, ""),
            )
            for idx, c in enumerate(self.active_cameras, 1)
        ]
        return PluginObservationContext(
            embodiment_summary=f"unseen 3D robot `{interp.robot_id}` (`{self.robosuite_robot}`)",
            task_instruction=self.task_instruction,
            turn_idx=turn_idx,
            max_turns=max_turns,
            allowed_units=units,
            camera_guides=cam_guides,
            arms=[
                ArmProprioPacket(
                    arm_label=interp.arm_names[0],
                    robot_id=interp.robot_id,
                    step_m=float(self.state.step_m),
                    joint_q_rad=q_rad,
                    tcp_base_xyz_m=[float(c) for c in tcp_b],
                    gripper_cmd=g_cmd,
                    gripper_encoder_qpos=g_qpos,
                    has_gripper=has_grip,
                )
            ],
            last_subtask=last_subtask,
            recent_history=list(self.recent_units),
            in_context_demos=list(in_context_demos or ()),
            is_bimanual=False,
            has_prev_visual_frame=has_prev_visual_frame,
        )

    def step_unit(self, unit: str) -> dict[str, Any]:
        arm = self.interpreter.arm_names[0]
        self.last_unit_label = unit
        tcp_before_base = self.proprio_tcp_in_base()
        self.state, g_res = self.interpreter.apply_unit(unit, self.state, arm=arm)
        self.step_count += 1
        self.recent_units.append(unit)

        q_target = self.state.arm_q[arm].copy()
        g_open = float(self.state.gripper_open[arm])
        if unit in ("GRASP", "RELEASE") and "robot0_gripper_qpos" in self.obs and self._env.action_dim > len(q_target):
            # Hardware-encoder-based gripper settling: handles both fast 2-finger jaws and slow 3-finger hands (e.g. Jaco speed=0.01)
            grip_speed = float(getattr(self._env.robots[0].gripper.get("right"), "speed", 0.1)) if hasattr(self._env.robots[0], "gripper") else 0.1
            max_grip_steps = 240 if grip_speed < 0.03 else 40
            rec_stride = 8 if max_grip_steps > 60 else 2
            prev_gq = np.asarray(self.obs["robot0_gripper_qpos"], dtype=np.float64).copy()
            still_cnt = 0
            for i in range(max_grip_steps):
                self._step_physics(q_target, g_open, record=(i % rec_stride == 0 or i == max_grip_steps - 1))
                cur_gq = np.asarray(self.obs["robot0_gripper_qpos"], dtype=np.float64)
                if float(np.max(np.abs(cur_gq - prev_gq))) < 2e-4:
                    still_cnt += 1
                    if i >= 33 and still_cnt >= 8:
                        self.video_frames.append(self.compose_frame_rgb())
                        break
                else:
                    still_cnt = 0
                prev_gq = cur_gq.copy()
        else:
            n_sub = 34 if unit in ("GRASP", "RELEASE") else (3 if unit.startswith("STEP_") else self.substeps_per_unit)
            for i in range(n_sub):
                self._step_physics(q_target, g_open, record=(i % 2 == 0 or i == n_sub - 1))

        self.state.arm_q[arm] = self.obs["robot0_joint_pos"].copy()
        actual_tf = self.interpreter.fk_tcp(self.state, arm=arm)
        if (
            unit in ("GRASP", "RELEASE")
            or g_res.clamped_by_workspace
            or float(np.linalg.norm(actual_tf.p - self.state.tcp_setpoint[arm].p)) > 0.008
        ):
            self.state.tcp_setpoint[arm] = actual_tf

        tcp_after_base = actual_tf.p.copy()
        g_cmd, g_qpos = self.raw_gripper_proprio()
        q_track_err = float(np.max(np.abs(self._joint_error(q_target, self.obs["robot0_joint_pos"]))))
        return {
            **g_res.to_dict(),
            "urdf_fk_tcp_before_base": [round(float(c), 4) for c in tcp_before_base],
            "urdf_fk_tcp_after_base": [round(float(c), 4) for c in tcp_after_base],
            "gripper_command": g_cmd,
            "gripper_encoder_qpos": g_qpos,
            "joint_tracking_max_err_rad": round(q_track_err, 5),
        }

    def check_success_audit_only(self) -> tuple[bool, dict[str, Any]]:
        """Post-turn evaluation oracle — NEVER shown to the VLM!"""
        ok = bool(self._env._check_success())
        info: dict[str, Any] = {"mujoco_task_success": ok}
        if "cubeA_pos" in self.obs and "cubeB_pos" in self.obs:
            ca = self.obs["cubeA_pos"]
            cb = self.obs["cubeB_pos"]
            info["audit_xy_err_m"] = round(float(np.linalg.norm(ca[:2] - cb[:2])), 4)
            info["audit_dz_m"] = round(float(ca[2] - cb[2]), 4)
        elif "hinge_qpos" in self.obs:
            hq = float(np.asarray(self.obs["hinge_qpos"]).reshape(-1)[0])
            info["door_hinge_rad"] = round(hq, 4)
            info["door_hinge_deg"] = round(math.degrees(hq), 2)
            if "handle_qpos" in self.obs:
                lq = float(np.asarray(self.obs["handle_qpos"]).reshape(-1)[0])
                info["latch_handle_rad"] = round(lq, 4)
        elif hasattr(self._env, "target_bin_placements") and hasattr(self._env, "object_id"):
            oid = int(self._env.object_id)
            oname = self._env.obj_names[oid]
            opos = self.obs.get(f"{oname}_pos")
            if opos is not None:
                slot_xy = np.asarray(self._env.target_bin_placements[oid])[:2]
                info["audit_xy_err_m"] = round(float(np.linalg.norm(np.asarray(opos)[:2] - slot_xy)), 4)
        elif self.robosuite_env_name.startswith("NutAssembly"):
            d = self._env.sim.data
            if "RoundNut_pos" in self.obs and self.robosuite_env_name != "NutAssemblySquare":
                c_pos = d.get_site_xpos("RoundNut_center_site")
                peg_pos = d.get_body_xpos("peg2")
                info["audit_xy_err_m"] = round(float(np.linalg.norm(c_pos[:2] - peg_pos[:2])), 4)
            elif "SquareNut_pos" in self.obs:
                c_pos = d.get_site_xpos("SquareNut_center_site")
                peg_pos = d.get_body_xpos("peg1")
                info["audit_xy_err_m"] = round(float(np.linalg.norm(c_pos[:2] - peg_pos[:2])), 4)
        elif hasattr(self._env, "wiped_markers") and hasattr(self._env, "num_markers"):
            total_m = max(1, int(self._env.num_markers))
            wiped_m = len(self._env.wiped_markers)
            info["wiped_markers"] = wiped_m
            info["total_markers"] = total_m
            info["wiped_ratio"] = round(wiped_m / total_m, 4)
        return ok, info

    def compose_frame_rgb(self) -> np.ndarray:
        """Compose a 1152x448 3-panel 3D MuJoCo frame with zero privileged metric overlays."""
        c1, c2, c3 = self.active_cameras
        img_1 = np.flipud(self.obs[f"{c1}_image"])
        img_2 = np.flipud(self.obs[f"{c2}_image"])
        img_3 = np.flipud(self.obs[f"{c3}_image"])

        W, H = 1152, 448
        canvas = Image.new("RGB", (W, H), (249, 247, 244))
        draw = ImageDraw.Draw(canvas)

        draw.rectangle((0, 0, W, 38), fill=(53, 61, 67))
        g_cmd, _ = self.raw_gripper_proprio()
        header_txt = (
            f"STRICT ZERO-PRIVILEGE (RAW RGB + ENCODERS)  |  {self.robosuite_env_name} · {self.interpreter.robot_id} ({self.robosuite_robot})  |  "
            f"Turn {self.current_turn_idx}  |  Last: {self.last_unit_label}  |  EE: {g_cmd}"
        )
        draw.text((14, 11), header_txt, fill=(255, 255, 255))

        panel_w, panel_h = 368, 368
        panels = [
            (12, img_1, (82, 107, 133), f"CAM 1: {_CAMERA_PANEL_TITLE.get(c1, c1.upper())}"),
            (392, img_2, (125, 104, 84), f"CAM 2: {_CAMERA_PANEL_TITLE.get(c2, c2.upper())}"),
            (772, img_3, (116, 99, 140), f"CAM 3: {_CAMERA_PANEL_TITLE.get(c3, c3.upper())}"),
        ]
        for x0, arr, col, title in panels:
            pil_p = Image.fromarray(arr).resize((panel_w, panel_h), Image.Resampling.BILINEAR)
            canvas.paste(pil_p, (x0, 46))
            draw.rectangle((x0, 46, x0 + panel_w, 46 + panel_h), outline=col, width=2)
            draw.rectangle((x0 + 4, 50, x0 + panel_w - 4, 72), fill=col)
            draw.text((x0 + 10, 54), title, fill=(255, 255, 255))

        draw.text(
            (14, 422),
            f"Task: {self.task_instruction[:78]}   |   Subtask: {self.current_subtask_label[:42]}",
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


@dataclass
class MuJoCoBimanualHarnessEnv:
    """Zero-privilege 3D MuJoCo HETEROGENEOUS BIMANUAL environment (two distinct unseen URDF robots!).

    Uses `env_configuration="parallel"` so both robots share the canonical +X forward / +Y left / +Z up
    mounting orientation:
      - `robot0` (mounted at `-Y`) -> semantic `right` arm (`right_chunk`)
      - `robot1` (mounted at `+Y`) -> semantic `left` arm (`left_chunk`)
    """

    robosuite_env_name: str
    right_robosuite_robot: str  # robot0 (-Y)
    left_robosuite_robot: str   # robot1 (+Y)
    right_interpreter: EmbodimentInterpreter
    left_interpreter: EmbodimentInterpreter
    task_instruction: str
    camera_size: int = 320
    substeps_per_unit: int = 14
    seed: int = 7
    env_kwargs: dict[str, Any] = field(default_factory=dict)

    _env: Any = field(init=False, repr=False)
    active_cameras: tuple[str, str, str, str] = (
        "sideview",
        "birdview",
        "robot0_eye_in_hand",
        "robot1_eye_in_hand",
    )
    obs: dict[str, Any] = field(init=False, repr=False)
    right_state: Any = field(init=False)
    left_state: Any = field(init=False)
    video_frames: list[np.ndarray] = field(default_factory=list, repr=False)
    step_count: int = 0
    recent_pairs: list[str] = field(default_factory=list)
    current_turn_idx: int = 0
    current_subtask_label: str = "initializing"
    last_action_label: str = "RESET"

    def __post_init__(self) -> None:
        import robosuite as suite
        from robosuite.controllers import load_part_controller_config
        from robosuite.controllers.composite.composite_controller_factory import (
            refactor_composite_controller_config,
        )

        arm_cfg = load_part_controller_config(default_controller="JOINT_POSITION")
        arm_cfg["kp"] = 280
        ctrl_cfgs = [
            refactor_composite_controller_config(arm_cfg, self.right_robosuite_robot, ["right"]),
            refactor_composite_controller_config(arm_cfg, self.left_robosuite_robot, ["right"]),
        ]
        self._env = suite.make(
            env_name=self.robosuite_env_name,
            robots=[self.right_robosuite_robot, self.left_robosuite_robot],
            env_configuration="parallel",
            controller_configs=ctrl_cfgs,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=list(self.active_cameras),
            camera_heights=self.camera_size,
            camera_widths=self.camera_size,
            control_freq=20,
            horizon=2500,
            **self.env_kwargs,
        )
        self.reset()

    @staticmethod
    def _reach_m(interp: EmbodimentInterpreter) -> float:
        tree = getattr(interp, "tree", None)
        if tree is not None:
            total = sum(float(np.linalg.norm(jt.origin.p)) for jt in tree.joints.values())
            if total > 0.2:
                return round(total + 0.05, 3)
        return 1.25

    def reset(self) -> None:
        np.random.seed(self.seed)
        self.obs = self._env.reset()

        for side_idx, (interp, attr, obs_key) in enumerate(
            (
                (self.right_interpreter, "right_state", "robot0_joint_pos"),
                (self.left_interpreter, "left_state", "robot1_joint_pos"),
            )
        ):
            st = interp.reset()
            arm = interp.arm_names[0]
            st.arm_q[arm] = self.obs[obs_key].copy()
            st.tcp_setpoint[arm] = interp.fk_tcp(st, arm=arm)
            st.gripper_open[arm] = 1.0
            st.step_m = 0.02
            reach = self._reach_m(interp)
            interp.workspace_bounds = (
                np.array([-reach, -reach, -reach], dtype=np.float64),
                np.array([reach, reach, reach], dtype=np.float64),
            )
            setattr(self, attr, st)

        self.video_frames.clear()
        self.step_count = 0
        self.recent_pairs.clear()
        self.current_turn_idx = 0
        self.current_subtask_label = "ready"
        self.last_action_label = "RESET"
        r_arm = self.right_interpreter.arm_names[0]
        l_arm = self.left_interpreter.arm_names[0]
        for _ in range(6):
            self._step_physics(
                self.right_state.arm_q[r_arm],
                1.0,
                self.left_state.arm_q[l_arm],
                1.0,
                record=True,
            )

    def _joint_error(self, robot_idx: int, q_target: np.ndarray, q_cur: np.ndarray) -> np.ndarray:
        diff = np.asarray(q_target, dtype=np.float64) - np.asarray(q_cur, dtype=np.float64)
        robot = self._env.robots[robot_idx]
        m = self._env.sim.model
        for i, jname in enumerate(robot.robot_joints):
            jid = m.joint_name2id(jname)
            if not bool(m.jnt_limited[jid]):
                diff[i] = (diff[i] + math.pi) % (2.0 * math.pi) - math.pi
        return diff

    def _step_physics(
        self,
        q_right: np.ndarray,
        g_right_open: float,
        q_left: np.ndarray,
        g_left_open: float,
        record: bool = True,
    ) -> None:
        a0 = np.clip(self._joint_error(0, q_right, self.obs["robot0_joint_pos"]) / 0.05, -1.0, 1.0)
        a1 = np.clip(self._joint_error(1, q_left, self.obs["robot1_joint_pos"]) / 0.05, -1.0, 1.0)
        g0 = -1.0 if g_right_open > 0.5 else 1.0
        g1 = -1.0 if g_left_open > 0.5 else 1.0
        act = np.concatenate([a0, [g0], a1, [g1]])
        self.obs, _, _, _ = self._env.step(act)
        if record:
            self.video_frames.append(self.compose_frame_rgb())

    def proprio_tcp_in_base(self, side: str) -> np.ndarray:
        if side == "right":
            arm = self.right_interpreter.arm_names[0]
            self.right_state.arm_q[arm] = self.obs["robot0_joint_pos"].copy()
            return self.right_interpreter.fk_tcp(self.right_state, arm=arm).p.copy()
        arm = self.left_interpreter.arm_names[0]
        self.left_state.arm_q[arm] = self.obs["robot1_joint_pos"].copy()
        return self.left_interpreter.fk_tcp(self.left_state, arm=arm).p.copy()

    def raw_gripper_proprio(self, side: str) -> tuple[str, list[float]]:
        if side == "right":
            arm = self.right_interpreter.arm_names[0]
            cmd = "OPEN" if float(self.right_state.gripper_open.get(arm, 1.0)) > 0.5 else "CLOSED"
            qpos = [round(float(v), 4) for v in self.obs.get("robot0_gripper_qpos", [])]
            return cmd, qpos
        arm = self.left_interpreter.arm_names[0]
        cmd = "OPEN" if float(self.left_state.gripper_open.get(arm, 1.0)) > 0.5 else "CLOSED"
        qpos = [round(float(v), 4) for v in self.obs.get("robot1_gripper_qpos", [])]
        return cmd, qpos

    def build_plugin_context(
        self,
        *,
        turn_idx: int,
        max_turns: int,
        last_subtask: str,
        in_context_demos: list[dict[str, Any]] | None,
        has_prev_visual_frame: bool = False,
    ) -> PluginObservationContext:
        units = valid_units_for_embodiment(supports_mobile_base=False)
        r_tcp = self.proprio_tcp_in_base("right")
        l_tcp = self.proprio_tcp_in_base("left")
        rg_cmd, rg_q = self.raw_gripper_proprio("right")
        lg_cmd, lg_q = self.raw_gripper_proprio("left")
        return PluginObservationContext(
            embodiment_summary=(
                f"a **HETEROGENEOUS UNSEEN BIMANUAL PAIR**: "
                f"LEFT arm (`left_chunk`) = `{self.left_interpreter.robot_id}` (`{self.left_robosuite_robot}`, mounted on robot's `+Y` side, foreground in CAM 1 side view and RIGHT side of CAM 2 overhead view), "
                f"RIGHT arm (`right_chunk`) = `{self.right_interpreter.robot_id}` (`{self.right_robosuite_robot}`, mounted on robot's `-Y` side, background in CAM 1 side view and LEFT side of CAM 2 overhead view)"
            ),
            task_instruction=self.task_instruction,
            turn_idx=turn_idx,
            max_turns=max_turns,
            allowed_units=units,
            camera_guides=[
                (
                    "CAM 1 (Top-Left — CLOSE-UP SIDE ELEVATION)",
                    _CAMERA_AXIS_GUIDE["sideview"]
                    + f" Foreground (closer) arm = LEFT arm (`{self.left_robosuite_robot}`, `left_chunk`, `+Y` side); background (farther) arm = RIGHT arm (`{self.right_robosuite_robot}`, `right_chunk`, `-Y` side). Use this close-up elevation view to judge exact finger-tip Z height relative to both handles before closing.",
                ),
                (
                    "CAM 2 (Top-Right — OVERHEAD TOP-DOWN)",
                    _CAMERA_AXIS_GUIDE["birdview"]
                    + f" LEFT side of image = RIGHT arm (`{self.right_robosuite_robot}`, `right_chunk`); RIGHT side of image = LEFT arm (`{self.left_robosuite_robot}`, `left_chunk`).",
                ),
                (
                    f"CAM 3 (Bottom-Left — RIGHT ARM WRIST: {self.right_robosuite_robot})",
                    _CAMERA_AXIS_GUIDE["robot0_eye_in_hand"],
                ),
                (
                    f"CAM 4 (Bottom-Right — LEFT ARM WRIST: {self.left_robosuite_robot})",
                    _CAMERA_AXIS_GUIDE["robot1_eye_in_hand"],
                ),
            ],
            arms=[
                ArmProprioPacket(
                    arm_label="left",
                    robot_id=self.left_interpreter.robot_id,
                    step_m=float(self.left_state.step_m),
                    joint_q_rad=[round(float(v), 4) for v in self.obs["robot1_joint_pos"]],
                    tcp_base_xyz_m=[float(c) for c in l_tcp],
                    gripper_cmd=lg_cmd,
                    gripper_encoder_qpos=lg_q,
                    has_gripper=True,
                    world_axis_note="controlled by `left_chunk` (foreground in CAM 1, RIGHT side of CAM 2)",
                ),
                ArmProprioPacket(
                    arm_label="right",
                    robot_id=self.right_interpreter.robot_id,
                    step_m=float(self.right_state.step_m),
                    joint_q_rad=[round(float(v), 4) for v in self.obs["robot0_joint_pos"]],
                    tcp_base_xyz_m=[float(c) for c in r_tcp],
                    gripper_cmd=rg_cmd,
                    gripper_encoder_qpos=rg_q,
                    has_gripper=True,
                    world_axis_note="controlled by `right_chunk` (background in CAM 1, LEFT side of CAM 2)",
                ),
            ],
            last_subtask=last_subtask,
            recent_history=list(self.recent_pairs),
            in_context_demos=list(in_context_demos or ()),
            is_bimanual=True,
            has_prev_visual_frame=has_prev_visual_frame,
            extra_embodiment_note=(
                "Both heterogeneous robots share identical parallel base axes: `+X` (`MV_FWD`) reaches toward the center/front of the table, "
                "`+Y` (`MV_LEFT`) moves right in CAM 2 overhead, `-Y` (`MV_RIGHT`) moves left in CAM 2 overhead, and `+Z` (`MV_UP`) lifts upward. "
                "Both parallel bases sit at the same table elevation, so symmetric handles sit at matching `base_link` Z height."
            ),
        )

    def step_bimanual_pair(self, left_unit: str, right_unit: str) -> dict[str, Any]:
        """Step BOTH unseen URDF interpreters synchronously in MuJoCo physics."""
        r_arm = self.right_interpreter.arm_names[0]
        l_arm = self.left_interpreter.arm_names[0]
        self.last_action_label = f"L:{left_unit} | R:{right_unit}"
        self.recent_pairs.append(self.last_action_label)
        self.step_count += 1

        q_r0 = self.obs["robot0_joint_pos"].copy()
        q_l0 = self.obs["robot1_joint_pos"].copy()

        self.right_state, r_res = self.right_interpreter.apply_unit(right_unit, self.right_state, arm=r_arm)
        self.left_state, l_res = self.left_interpreter.apply_unit(left_unit, self.left_state, arm=l_arm)

        q_r = self.right_state.arm_q[r_arm].copy()
        g_r = float(self.right_state.gripper_open[r_arm])
        q_l = self.left_state.arm_q[l_arm].copy()
        g_l = float(self.left_state.gripper_open[l_arm])

        is_grip = left_unit in ("GRASP", "RELEASE") or right_unit in ("GRASP", "RELEASE")
        is_step_only = (left_unit.startswith("STEP_") or left_unit == "HOLD") and (
            right_unit.startswith("STEP_") or right_unit == "HOLD"
        )
        n_sub = 34 if is_grip else (3 if is_step_only else self.substeps_per_unit)
        n_ramp = max(1, n_sub - 5)
        dq_r = self._joint_error(0, q_r, q_r0)
        dq_l = self._joint_error(1, q_l, q_l0)
        for i in range(n_sub):
            alpha = 1.0 if (is_grip or is_step_only) else min(1.0, (i + 1) / n_ramp)
            self._step_physics(
                q_r0 + alpha * dq_r,
                g_r,
                q_l0 + alpha * dq_l,
                g_l,
                record=(i % 2 == 0 or i == n_sub - 1),
            )

        # Encoder anti-windup per arm (threshold 0.04 m avoids erasing coordinated lift under payload sag)
        self.right_state.arm_q[r_arm] = self.obs["robot0_joint_pos"].copy()
        r_tf = self.right_interpreter.fk_tcp(self.right_state, arm=r_arm)
        if right_unit in ("GRASP", "RELEASE") or r_res.clamped_by_workspace or float(np.linalg.norm(r_tf.p - self.right_state.tcp_setpoint[r_arm].p)) > 0.04:
            self.right_state.tcp_setpoint[r_arm] = r_tf

        self.left_state.arm_q[l_arm] = self.obs["robot1_joint_pos"].copy()
        l_tf = self.left_interpreter.fk_tcp(self.left_state, arm=l_arm)
        if left_unit in ("GRASP", "RELEASE") or l_res.clamped_by_workspace or float(np.linalg.norm(l_tf.p - self.left_state.tcp_setpoint[l_arm].p)) > 0.04:
            self.left_state.tcp_setpoint[l_arm] = l_tf

        return {
            "left": {**l_res.to_dict(), "urdf_fk_tcp_after_base": [round(float(c), 4) for c in l_tf.p]},
            "right": {**r_res.to_dict(), "urdf_fk_tcp_after_base": [round(float(c), 4) for c in r_tf.p]},
        }

    def check_success_audit_only(self) -> tuple[bool, dict[str, Any]]:
        """Post-turn evaluation oracle — NEVER shown to the VLM!"""
        ok = bool(self._env._check_success())
        info: dict[str, Any] = {"mujoco_task_success": ok}
        if hasattr(self._env, "pot_center_id") and hasattr(self._env, "table_top_id"):
            pot_bot = float(self._env.sim.data.site_xpos[self._env.pot_center_id][2] - self._env.pot.top_offset[2])
            tbl_z = float(self._env.sim.data.site_xpos[self._env.table_top_id][2])
            info["pot_lift_clearance_m"] = round(pot_bot - tbl_z, 4)
        return ok, info

    def compose_frame_rgb(self) -> np.ndarray:
        """Compose a crisp 768x800 2x2 square-aspect 4-camera heterogeneous bimanual frame."""
        c1, c2, c3, c4 = self.active_cameras
        i1 = np.flipud(self.obs[f"{c1}_image"])
        i2 = np.flipud(self.obs[f"{c2}_image"])
        i3 = np.flipud(self.obs[f"{c3}_image"])
        i4 = np.flipud(self.obs[f"{c4}_image"])

        W, H = 768, 800
        canvas = Image.new("RGB", (W, H), (249, 247, 244))
        draw = ImageDraw.Draw(canvas)

        draw.rectangle((0, 0, W, 38), fill=(53, 61, 67))
        lg_cmd, _ = self.raw_gripper_proprio("left")
        rg_cmd, _ = self.raw_gripper_proprio("right")
        header = (
            f"ZERO-PRIVILEGE HETEROGENEOUS BIMANUAL  |  {self.robosuite_env_name} · "
            f"L:{self.left_robosuite_robot} + R:{self.right_robosuite_robot}  |  "
            f"Turn {self.current_turn_idx}  |  L:{lg_cmd} R:{rg_cmd}"
        )
        draw.text((14, 11), header, fill=(255, 255, 255))

        pw, ph = 366, 360
        panels = [
            (12, 46, i1, (82, 107, 133), "CAM 1: SIDE ELEVATION (BOTH ARMS)"),
            (390, 46, i2, (125, 104, 84), "CAM 2: OVERHEAD TOP-DOWN"),
            (12, 414, i3, (150, 92, 102), f"CAM 3: RIGHT ARM WRIST ({self.right_robosuite_robot})"),
            (390, 414, i4, (116, 99, 140), f"CAM 4: LEFT ARM WRIST ({self.left_robosuite_robot})"),
        ]
        for x0, y0, arr, col, title in panels:
            pil_p = Image.fromarray(arr).resize((pw, ph), Image.Resampling.BILINEAR)
            canvas.paste(pil_p, (x0, y0))
            draw.rectangle((x0, y0, x0 + pw, y0 + ph), outline=col, width=2)
            draw.rectangle((x0 + 4, y0 + 4, x0 + pw - 4, y0 + 26), fill=col)
            draw.text((x0 + 10, y0 + 8), title, fill=(255, 255, 255))

        draw.text(
            (14, 780),
            f"Last: {self.last_action_label}   |   Subtask: {self.current_subtask_label[:54]}",
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


def _ensure_synthesized_embodiment(
    *,
    robosuite_robot: str,
    robot_id: str,
    env_name: str,
    backend: FrontierBackend,
    rroot: Path,
    hroot: Path,
    force_resynthesize: bool,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    urdf_path = rroot / robot_id / "robot.urdf"
    if force_resynthesize or not urdf_path.is_file():
        export_meta = _export_urdf_subprocess(robosuite_robot, robot_id, urdf_path, env_name=env_name)
    else:
        tree_cached = parse_robot_file(urdf_path)
        tcp_cand = f"{robot_id}_tcp" if f"{robot_id}_tcp" in tree_cached.links else tree_cached.root_link
        export_meta = {
            "robosuite_robot": robosuite_robot,
            "robot_id": robot_id,
            "source_env": env_name,
            "urdf_path": urdf_path.name,
            "actuated_joints": [
                jn
                for jn, jt in tree_cached.joints.items()
                if jt.joint_type in ("revolute", "continuous") and jt.mimic_joint is None
            ],
            "tcp_link": tcp_cand,
            "mujoco_fk_agreement_err_m": 0.0,
        }

    interp_path = hroot / "embodiments" / f"{robot_id}.py"
    common_path = hroot / "common" / "urdf_interpreter_base.py"
    synth = URDFHarnessSynthesizer(backend=backend, max_attempts=3)
    if force_resynthesize or not interp_path.is_file():
        synth_rep = synth.synthesize_embodiment_interpreter(
            urdf_path=urdf_path,
            output_py_path=interp_path,
            sidecar_path=None,
            common_library_path=common_path if common_path.is_file() else None,
            record_full_trace=True,
        )
        if not synth_rep.passed:
            raise RuntimeError(f"Interpreter synthesis failed on {robot_id}: {synth_rep.verification}")
        synth_dict = synth_rep.to_dict()
    else:
        ver_tmp = verify_interpreter_isolated(interp_path, urdf_path, harnesses_root=hroot)
        # Preserve original Stage-1 synthesis trace if previously recorded for this robot_id
        prior_turns: list[dict[str, Any]] = []
        prior_model = backend.model_name
        prior_tokens = 0
        prior_elapsed = 0.0
        prior_attempts = 1
        live_dir = hroot.parent / "artifacts" / "live_runs"
        if live_dir.is_dir():
            for cand_f in sorted(live_dir.glob("unseen_mujoco_*.json")):
                try:
                    cd = json.loads(cand_f.read_text(encoding="utf-8"))
                    if cd.get("robot_id") == robot_id:
                        sr = cd.get("stage1_urdf_and_synthesis", {}).get("synthesis_report", {})
                        if sr.get("turns"):
                            prior_turns = list(sr["turns"])
                            prior_model = str(sr.get("model", prior_model))
                            prior_tokens = int(sr.get("total_tokens", 0))
                            prior_elapsed = float(sr.get("total_elapsed_s", 0.0))
                            prior_attempts = int(sr.get("attempts", len(prior_turns)))
                            break
                except Exception:
                    continue
        synth_dict = {
            "robot_id": robot_id,
            "model": prior_model,
            "passed": ver_tmp.passed,
            "attempts": prior_attempts,
            "total_tokens": prior_tokens,
            "total_elapsed_s": prior_elapsed,
            "verification": ver_tmp.to_dict(),
            "turns": prior_turns,
        }
    return export_meta, synth_dict, urdf_path, interp_path


def run_unseen_mujoco_pipeline(
    *,
    run_id: str,
    robosuite_robot: str,
    robot_id: str,
    robosuite_env_name: str,
    task_instruction: str,
    backend: FrontierBackend,
    robots_root: str | Path,
    harnesses_root: str | Path,
    media_out_dir: str | Path,
    trace_json_out: str | Path,
    priors_path: str | Path | None = None,
    seed: int = 7,
    max_turns: int = 20,
    force_resynthesize: bool = True,
    env_kwargs: dict[str, Any] | None = None,
    plugin_cfg: HarnessPluginConfig | None = None,
) -> dict[str, Any]:
    """Zero-Privilege Single-Arm Unseen Embodiment 3D MuJoCo Pipeline."""
    t_start = time.monotonic()
    rroot = Path(robots_root)
    hroot = Path(harnesses_root)
    mdir = Path(media_out_dir) / run_id
    if mdir.exists():
        for old_f in mdir.glob("*"):
            if old_f.is_file():
                old_f.unlink()
    mdir.mkdir(parents=True, exist_ok=True)

    export_meta, synth_dict, urdf_path, interp_path = _ensure_synthesized_embodiment(
        robosuite_robot=robosuite_robot,
        robot_id=robot_id,
        env_name=robosuite_env_name,
        backend=backend,
        rroot=rroot,
        hroot=hroot,
        force_resynthesize=force_resynthesize,
    )

    interp = load_interpreter_module(interp_path, urdf_path, harnesses_root=hroot)
    env = MuJoCoPhysicalHarnessEnv(
        robosuite_env_name=robosuite_env_name,
        robosuite_robot=robosuite_robot,
        interpreter=interp,
        task_instruction=task_instruction,
        seed=seed,
        env_kwargs=env_kwargs or {},
    )

    demos = None
    if priors_path and Path(priors_path).is_file():
        demos = SemanticPriorBank.load_json(priors_path).retrieve(task_instruction, top_k=2)

    pcfg = plugin_cfg or HarnessPluginConfig()
    pipeline = PluginPipeline(cfg=pcfg)
    schema = single_arm_step_json_schema(supports_mobile_base=interp.supports_mobile_base)
    vlm_turns: list[dict[str, Any]] = []
    last_subtask = ""
    prev_obs_jpg: Path | None = None
    vlm_total_tokens = 0
    vlm_total_elapsed = 0.0

    try:
        for turn_idx in range(1, max_turns + 1):
            env.current_turn_idx = turn_idx
            obs_jpg = env.save_observation_jpg(mdir / f"turn_{turn_idx:02d}.jpg")
            use_diff = bool(pcfg.visual_diff_frame and prev_obs_jpg is not None)
            ctx = env.build_plugin_context(
                turn_idx=turn_idx,
                max_turns=max_turns,
                last_subtask=last_subtask,
                in_context_demos=demos,
                has_prev_visual_frame=use_diff,
            )
            prompt = pipeline.build_prompt(ctx)
            images = [prev_obs_jpg, obs_jpg] if (use_diff and prev_obs_jpg is not None) else [obs_jpg]
            resp = backend.generate(prompt=prompt, images=images, json_schema=schema, timeout_s=480.0)
            vlm_total_tokens += resp.tokens_used
            vlm_total_elapsed += resp.elapsed_s

            parsed = resp.parse_json()
            thought = str(parsed.get("thought", ""))
            subtask = str(parsed.get("current_subtask", ""))
            chunk = [str(u) for u in parsed.get("action_chunk", ["HOLD"])]
            last_subtask = subtask
            env.current_subtask_label = subtask

            unit_logs: list[dict[str, Any]] = []
            done_emitted = False
            for u in chunk:
                if u == "DONE":
                    done_emitted = True
                    unit_logs.append({"unit": "DONE", "accepted": True, "note": "vlm_declared_done"})
                    break
                unit_logs.append(env.step_unit(u))

            prev_obs_jpg = obs_jpg
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
                arm = interp.arm_names[0]
                for _ in range(14):
                    env._step_physics(env.state.arm_q[arm], env.state.gripper_open[arm], record=True)
                env.save_observation_jpg(mdir / "final_success.jpg")
                break
            if done_emitted and not ok_now:
                last_subtask = (
                    "Previous DONE was rejected (goal condition not yet fully met). "
                    "Do NOT emit DONE again without first taking physical corrective action "
                    "(e.g., lift clearly higher, sweep remaining marks, seat deeper, or refine alignment)."
                )

        ok_final, final_audit = env.check_success_audit_only()
        env.export_mp4(mdir / "rollout.mp4", fps=20)
        num_video_frames = len(env.video_frames)
    finally:
        env.close()

    final_metrics: dict[str, Any] = {"mujoco_task_success": final_audit["mujoco_task_success"]}
    if "audit_xy_err_m" in final_audit:
        final_metrics["xy_alignment_err_m"] = final_audit["audit_xy_err_m"]
    if "audit_dz_m" in final_audit:
        final_metrics["red_above_green_dz_m"] = final_audit["audit_dz_m"]
    if "door_hinge_deg" in final_audit:
        final_metrics["door_hinge_deg"] = final_audit["door_hinge_deg"]
        final_metrics["door_hinge_rad"] = final_audit["door_hinge_rad"]
    if "latch_handle_rad" in final_audit:
        final_metrics["latch_handle_rad"] = final_audit["latch_handle_rad"]
    if "wiped_markers" in final_audit:
        final_metrics["wiped_markers"] = final_audit["wiped_markers"]
        final_metrics["total_markers"] = final_audit["total_markers"]
        final_metrics["wiped_ratio"] = final_audit["wiped_ratio"]

    payload = sanitize_payload(
        {
            "run_id": run_id,
            "zero_privilege": True,
            "is_bimanual": False,
            "task_name": robosuite_env_name,
            "robot_id": robot_id,
            "robosuite_robot": robosuite_robot,
            "environment": f"MuJoCo 3D · robosuite {robosuite_env_name} (Zero-Privilege RGB + Proprio Only)",
            "unseen_embodiment": True,
            "sidecar_used": False,
            "plugins_enabled": pcfg.enabled_plugin_names(),
            "task_instruction": task_instruction,
            "model": backend.model_name,
            "reasoning_effort": getattr(backend, "reasoning_effort", "max"),
            "fast_mode": bool(getattr(backend, "fast_mode", False)),
            "success": ok_final,
            "video_path": f"media/{run_id}/rollout.mp4",
            "video_frames": num_video_frames,
            "video_fps": 20,
            "video_duration_s": round(num_video_frames / 20.0, 2),
            "total_vlm_turns": len(vlm_turns),
            "total_semantic_units": sum(len(t["action_chunk"]) for t in vlm_turns),
            "total_vlm_tokens": vlm_total_tokens,
            "total_vlm_elapsed_s": round(vlm_total_elapsed, 3),
            "total_pipeline_elapsed_s": round(time.monotonic() - t_start, 3),
            "final_metrics": final_metrics,
            "stage1_urdf_and_synthesis": {
                "urdf_export": export_meta,
                "urdf_xml": urdf_path.read_text(encoding="utf-8"),
                "synthesized_interpreter_code": interp_path.read_text(encoding="utf-8"),
                "synthesis_report": synth_dict,
            },
            "stage2_vlm_closed_loop_turns": vlm_turns,
        }
    )

    out_json = Path(trace_json_out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def run_bimanual_mujoco_pipeline(
    *,
    run_id: str,
    right_robosuite_robot: str,
    right_robot_id: str,
    left_robosuite_robot: str,
    left_robot_id: str,
    robosuite_env_name: str,
    task_instruction: str,
    backend: FrontierBackend,
    robots_root: str | Path,
    harnesses_root: str | Path,
    media_out_dir: str | Path,
    trace_json_out: str | Path,
    priors_path: str | Path | None = None,
    seed: int = 11,
    max_turns: int = 24,
    force_resynthesize: bool = False,
    plugin_cfg: HarnessPluginConfig | None = None,
) -> dict[str, Any]:
    """Zero-Privilege Heterogeneous Unseen-URDF Bimanual 3D MuJoCo Pipeline (`left_chunk` + `right_chunk`)."""
    t_start = time.monotonic()
    rroot = Path(robots_root)
    hroot = Path(harnesses_root)
    mdir = Path(media_out_dir) / run_id
    if mdir.exists():
        for old_f in mdir.glob("*"):
            if old_f.is_file():
                old_f.unlink()
    mdir.mkdir(parents=True, exist_ok=True)

    r_exp, r_syn, r_urdf, r_py = _ensure_synthesized_embodiment(
        robosuite_robot=right_robosuite_robot,
        robot_id=right_robot_id,
        env_name="Lift",
        backend=backend,
        rroot=rroot,
        hroot=hroot,
        force_resynthesize=force_resynthesize,
    )
    l_exp, l_syn, l_urdf, l_py = _ensure_synthesized_embodiment(
        robosuite_robot=left_robosuite_robot,
        robot_id=left_robot_id,
        env_name="Lift",
        backend=backend,
        rroot=rroot,
        hroot=hroot,
        force_resynthesize=force_resynthesize,
    )

    r_interp = load_interpreter_module(r_py, r_urdf, harnesses_root=hroot)
    l_interp = load_interpreter_module(l_py, l_urdf, harnesses_root=hroot)

    env = MuJoCoBimanualHarnessEnv(
        robosuite_env_name=robosuite_env_name,
        right_robosuite_robot=right_robosuite_robot,
        left_robosuite_robot=left_robosuite_robot,
        right_interpreter=r_interp,
        left_interpreter=l_interp,
        task_instruction=task_instruction,
        seed=seed,
    )

    demos = None
    if priors_path and Path(priors_path).is_file():
        demos = SemanticPriorBank.load_json(priors_path).retrieve(task_instruction, top_k=2)

    pcfg = plugin_cfg or HarnessPluginConfig()
    pipeline = PluginPipeline(cfg=pcfg)
    schema = bimanual_step_json_schema(supports_mobile_base=False)
    vlm_turns: list[dict[str, Any]] = []
    last_subtask = ""
    prev_obs_jpg: Path | None = None
    vlm_total_tokens = 0
    vlm_total_elapsed = 0.0

    try:
        for turn_idx in range(1, max_turns + 1):
            env.current_turn_idx = turn_idx
            obs_jpg = env.save_observation_jpg(mdir / f"turn_{turn_idx:02d}.jpg")
            use_diff = bool(pcfg.visual_diff_frame and prev_obs_jpg is not None)
            ctx = env.build_plugin_context(
                turn_idx=turn_idx,
                max_turns=max_turns,
                last_subtask=last_subtask,
                in_context_demos=demos,
                has_prev_visual_frame=use_diff,
            )
            prompt = pipeline.build_prompt(ctx)
            images = [prev_obs_jpg, obs_jpg] if (use_diff and prev_obs_jpg is not None) else [obs_jpg]
            resp = backend.generate(prompt=prompt, images=images, json_schema=schema, timeout_s=480.0)
            vlm_total_tokens += resp.tokens_used
            vlm_total_elapsed += resp.elapsed_s

            parsed = resp.parse_json()
            thought = str(parsed.get("thought", ""))
            subtask = str(parsed.get("current_subtask", ""))
            l_chunk = [str(u) for u in parsed.get("left_chunk", ["HOLD"])]
            r_chunk = [str(u) for u in parsed.get("right_chunk", ["HOLD"])]
            last_subtask = subtask
            env.current_subtask_label = subtask

            unit_logs: list[dict[str, Any]] = []
            done_emitted = False
            n_steps = max(len(l_chunk), len(r_chunk))
            for idx in range(n_steps):
                lu = l_chunk[idx] if idx < len(l_chunk) else "HOLD"
                ru = r_chunk[idx] if idx < len(r_chunk) else "HOLD"
                if lu == "DONE" and ru == "DONE":
                    done_emitted = True
                    unit_logs.append({"left": "DONE", "right": "DONE", "note": "vlm_declared_done"})
                    break
                if lu == "DONE":
                    lu = "HOLD"
                if ru == "DONE":
                    ru = "HOLD"
                unit_logs.append(env.step_bimanual_pair(lu, ru))

            prev_obs_jpg = obs_jpg
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
                    "left_chunk": l_chunk,
                    "right_chunk": r_chunk,
                    "action_chunk": [f"L:{u}" for u in l_chunk] + [f"R:{u}" for u in r_chunk],
                    "token_usage": sanitize_payload(resp.raw_meta),
                    "tokens_used": resp.tokens_used,
                    "elapsed_s": resp.elapsed_s,
                    "unit_grounding_logs": unit_logs,
                    "post_turn_audit_only": audit_metrics,
                }
            )

            if ok_now:
                ra = r_interp.arm_names[0]
                la = l_interp.arm_names[0]
                for _ in range(14):
                    env._step_physics(
                        env.right_state.arm_q[ra],
                        env.right_state.gripper_open[ra],
                        env.left_state.arm_q[la],
                        env.left_state.gripper_open[la],
                        record=True,
                    )
                env.save_observation_jpg(mdir / "final_success.jpg")
                break
            if done_emitted and not ok_now:
                last_subtask = (
                    "Previous DONE was rejected (goal condition not yet fully met). "
                    "Do NOT repeat DONE without first taking physical corrective action "
                    "(e.g., lift BOTH arms noticeably higher together or verify firm symmetric grasp)."
                )

        ok_final, final_audit = env.check_success_audit_only()
        env.export_mp4(mdir / "rollout.mp4", fps=20)
        num_video_frames = len(env.video_frames)
    finally:
        env.close()

    final_metrics: dict[str, Any] = {"mujoco_task_success": final_audit["mujoco_task_success"]}
    if "pot_lift_clearance_m" in final_audit:
        final_metrics["pot_lift_clearance_m"] = final_audit["pot_lift_clearance_m"]

    combined_robot_id = f"{left_robot_id}+{right_robot_id}"
    combined_robot_label = f"{left_robosuite_robot} + {right_robosuite_robot}"
    combined_urdf_preview = (
        f"<!-- LEFT ARM ({left_robot_id}) -->\n{l_urdf.read_text(encoding='utf-8')}\n"
        f"<!-- RIGHT ARM ({right_robot_id}) -->\n{r_urdf.read_text(encoding='utf-8')}"
    )
    combined_code_preview = (
        f"# === LEFT ARM INTERPRETER ({left_robot_id}.py) ===\n{l_py.read_text(encoding='utf-8')}\n\n"
        f"# === RIGHT ARM INTERPRETER ({right_robot_id}.py) ===\n{r_py.read_text(encoding='utf-8')}"
    )

    payload = sanitize_payload(
        {
            "run_id": run_id,
            "zero_privilege": True,
            "is_bimanual": True,
            "task_name": robosuite_env_name,
            "robot_id": combined_robot_id,
            "robosuite_robot": combined_robot_label,
            "environment": f"MuJoCo 3D · robosuite {robosuite_env_name} Heterogeneous Bimanual (Zero-Privilege 4-Cam RGB + Encoders)",
            "unseen_embodiment": True,
            "sidecar_used": False,
            "plugins_enabled": pcfg.enabled_plugin_names(),
            "task_instruction": task_instruction,
            "model": backend.model_name,
            "reasoning_effort": getattr(backend, "reasoning_effort", "max"),
            "fast_mode": bool(getattr(backend, "fast_mode", False)),
            "success": ok_final,
            "video_path": f"media/{run_id}/rollout.mp4",
            "video_frames": num_video_frames,
            "video_fps": 20,
            "video_duration_s": round(num_video_frames / 20.0, 2),
            "total_vlm_turns": len(vlm_turns),
            "total_semantic_units": sum(len(t["left_chunk"]) + len(t["right_chunk"]) for t in vlm_turns),
            "total_vlm_tokens": vlm_total_tokens,
            "total_vlm_elapsed_s": round(vlm_total_elapsed, 3),
            "total_pipeline_elapsed_s": round(time.monotonic() - t_start, 3),
            "final_metrics": final_metrics,
            "stage1_urdf_and_synthesis": {
                "urdf_export": {"left": l_exp, "right": r_exp},
                "urdf_xml": combined_urdf_preview,
                "urdf_xml_by_arm": {
                    "left": l_urdf.read_text(encoding="utf-8"),
                    "right": r_urdf.read_text(encoding="utf-8"),
                },
                "synthesized_interpreter_code": combined_code_preview,
                "synthesis_report": {
                    "robot_id": combined_robot_id,
                    "model": backend.model_name,
                    "passed": bool(l_syn.get("passed", True) and r_syn.get("passed", True)),
                    "attempts": int(l_syn.get("attempts", 1)) + int(r_syn.get("attempts", 1)),
                    "total_tokens": int(l_syn.get("total_tokens", 0)) + int(r_syn.get("total_tokens", 0)),
                    "turns": list(l_syn.get("turns", [])) + list(r_syn.get("turns", [])),
                    "verification": {"left": l_syn.get("verification", {}), "right": r_syn.get("verification", {})},
                },
            },
            "stage2_vlm_closed_loop_turns": vlm_turns,
        }
    )

    out_json = Path(trace_json_out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
