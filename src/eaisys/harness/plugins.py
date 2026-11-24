"""Ablation-grade Perceive -> Reason -> Act harness plugins (100% strict zero-privilege).

Design principles:
- One toggle per plugin in `HarnessPluginConfig`; when disabled, that section is omitted byte-identically.
- Unified across 3D single-arm MuJoCo, 3D heterogeneous bimanual MuJoCo, and offline unit-test envs.
- Enforces `assert_zero_privilege_prompt()` at runtime so privileged simulator leaks fail fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Protocol

from eaisys.harness.action_space import valid_units_for_embodiment
from eaisys.memory.playbook import PlaybookStore


class ZeroPrivilegeViolationError(RuntimeError):
    """Raised if a prompt accidentally contains privileged simulator state or cheat hints."""


_FORBIDDEN_PRIVILEGE_TOKENS: tuple[str, ...] = (
    "cubeA_pos",
    "cubeB_pos",
    "red_cube_xyz",
    "green_cube_xyz",
    "Scene `red_block`",
    "Scene `blue_plate`",
    "handle_pos",
    "distance_to_goal",
    "Height above table",
    "height above table",
    "above table surface",
    "clearance above table",
    "table surface height",
    "table_z",
    "table_offset",
    "CLOSED_HOLDING_OBJECT",
    "CLOSED_EMPTY_AIR",
    "Gripper: HOLDING",
    "grid squares",
    "gridlines",
    "per cell",
    "u ≈",
    "v ≈",
)
_CM_REGEX = re.compile(r"\b\d+(?:\.\d+)?\s*cm\b", re.IGNORECASE)


def assert_zero_privilege_prompt(prompt: str) -> None:
    """Runtime static audit guaranteeing zero simulator object/table/ruler/oracle privilege in prompts."""
    for tok in _FORBIDDEN_PRIVILEGE_TOKENS:
        if tok in prompt:
            raise ZeroPrivilegeViolationError(f"Privileged token {tok!r} detected in VLM prompt!")
    harness_lines = "\n".join(
        ln for ln in prompt.splitlines() if not ln.lstrip().startswith("- Previous subtask:")
    )
    m = _CM_REGEX.search(harness_lines)
    if m:
        raise ZeroPrivilegeViolationError(f"Privileged metric measurement {m.group(0)!r} detected in VLM prompt!")


@dataclass
class HarnessPluginConfig:
    """Ablation-grade toggles for every Perceive-Reason-Act plugin."""

    multi_view_guidance: bool = True
    proprioception_feedback: bool = True
    subtask_planning: bool = True
    adaptive_step_hint: bool = True
    action_chunking: bool = True
    action_history_window: int = 12
    cross_embodiment_priors: bool = True
    playbook_memory: bool = True
    playbook_version: str = "v1"
    visual_diff_frame: bool = False

    def enabled_plugin_names(self) -> list[str]:
        names: list[str] = []
        if self.multi_view_guidance:
            names.append("multi_view_guidance")
        if self.proprioception_feedback:
            names.append("proprioception_feedback")
        if self.subtask_planning:
            names.append("subtask_planning")
        if self.adaptive_step_hint:
            names.append("adaptive_step_hint")
        if self.action_chunking:
            names.append("action_chunking")
        if self.action_history_window > 0:
            names.append(f"action_history(w={self.action_history_window})")
        if self.cross_embodiment_priors:
            names.append("cross_embodiment_priors")
        if self.playbook_memory:
            names.append(f"playbook_memory({self.playbook_version})")
        if self.visual_diff_frame:
            names.append("visual_diff_frame")
        return names


@dataclass
class ArmProprioPacket:
    arm_label: str
    robot_id: str
    step_m: float
    joint_q_rad: list[float]
    tcp_base_xyz_m: list[float]
    gripper_cmd: str
    gripper_encoder_qpos: list[float]
    has_gripper: bool = True
    world_axis_note: str = ""


@dataclass
class PluginObservationContext:
    """Strictly zero-privilege observation packet consumed by all HarnessPlugins."""

    embodiment_summary: str
    task_instruction: str
    turn_idx: int
    max_turns: int
    allowed_units: list[str]
    camera_guides: list[tuple[str, str]]  # [(title, axis_guide), ...]
    arms: list[ArmProprioPacket]
    last_subtask: str = ""
    recent_history: list[Any] = field(default_factory=list)
    in_context_demos: list[dict[str, Any]] = field(default_factory=list)
    is_bimanual: bool = False
    has_prev_visual_frame: bool = False
    extra_embodiment_note: str = ""


class HarnessPlugin(Protocol):
    name: str

    def render(self, ctx: PluginObservationContext, cfg: HarnessPluginConfig) -> str | None: ...


class MultiViewGuidancePlugin:
    name = "multi_view_guidance"

    def render(self, ctx: PluginObservationContext, cfg: HarnessPluginConfig) -> str | None:
        if not cfg.multi_view_guidance or not ctx.camera_guides:
            return None
        lines = [
            f"{idx}. **{title}**: {guide}"
            for idx, (title, guide) in enumerate(ctx.camera_guides, 1)
        ]
        if cfg.visual_diff_frame and ctx.has_prev_visual_frame:
            lines.append(
                "- **Visual Progress Diff**: Two images are attached — Image 1 is the PREVIOUS turn frame (`t-1`) "
                "and Image 2 is the CURRENT live frame (`t`). Compare them to confirm physical motion and contact effects."
            )
        return "## Multi-View Camera Orientation (Robot Base Axes Only)\n" + "\n".join(lines)


class ProprioceptionPlugin:
    name = "proprioception_feedback"

    def render(self, ctx: PluginObservationContext, cfg: HarnessPluginConfig) -> str | None:
        if not cfg.proprioception_feedback or not ctx.arms:
            return None
        lines: list[str] = []
        for a in ctx.arms:
            gear = (
                "STEP_COARSE"
                if a.step_m >= 0.035
                else ("STEP_FINE" if a.step_m >= 0.015 else "STEP_ULTRA_FINE")
            )
            grip_str = (
                f"gripper cmd: `{a.gripper_cmd}` | raw jaw encoder: `{a.gripper_encoder_qpos}`"
                if a.has_gripper
                else f"end-effector tool: `{a.gripper_cmd}` (fixed non-prehensile tool)"
            )
            axis_suffix = f" | {a.world_axis_note}" if a.world_axis_note else ""
            lines.append(
                f"- Arm `{a.arm_label}` (`{a.robot_id}`): gear `{gear}` (`{a.step_m:.2f} m`) | "
                f"joint encoders `q` (rad): `{a.joint_q_rad}` | "
                f"URDF FK TCP in `base_link` (m): `[{a.tcp_base_xyz_m[0]:.4f}, {a.tcp_base_xyz_m[1]:.4f}, {a.tcp_base_xyz_m[2]:.4f}]` | "
                f"{grip_str}{axis_suffix}"
            )
        return "## Robot Hardware Encoders & URDF Forward Kinematics (`base_link` Frame Only)\n" + "\n".join(lines)


class AdaptiveStepPlugin:
    name = "adaptive_step_hint"

    def render(self, ctx: PluginObservationContext, cfg: HarnessPluginConfig) -> str | None:
        if not cfg.adaptive_step_hint:
            return None
        if ctx.is_bimanual:
            return (
                "## Adaptive Step & Bimanual Visual Verification\n"
                "- Output synchronized `left_chunk` (for `left` arm) and `right_chunk` (for `right` arm) of 1 to 5 units each.\n"
                "- Switch between `STEP_COARSE`, `STEP_FINE`, and `STEP_ULTRA_FINE` independently per arm based on visual proximity.\n"
                "- Cross-check BOTH arms across all camera panels before simultaneous `GRASP` and synchronized `MV_UP` lifting."
            )
        return (
            "## Adaptive Step & Visual Verification\n"
            "- Cross-check 3D alignment across all camera panels before contact, grasping, insertion, wiping, or pulling.\n"
            "- Use `STEP_COARSE` for longer free-space traversals and `STEP_FINE` / `STEP_ULTRA_FINE` (plus `ROT_*` when needed) for fine alignment.\n"
            "- Visually confirm every physical state change in the camera frames before advancing to the next stage or emitting `DONE`."
        )


class SubtaskPlanningPlugin:
    name = "subtask_planning"

    def render(self, ctx: PluginObservationContext, cfg: HarnessPluginConfig) -> str | None:
        if not cfg.subtask_planning:
            return None
        return f"## Subtask Context\n- Previous subtask: `{ctx.last_subtask or 'not_started'}`"


class ActionHistoryPlugin:
    name = "action_history"

    def render(self, ctx: PluginObservationContext, cfg: HarnessPluginConfig) -> str | None:
        if cfg.action_history_window <= 0:
            return None
        hist = ctx.recent_history[-cfg.action_history_window :]
        return f"## Recent Action History\n- Last executed units: `{hist if hist else '[]'}`"


class PlaybookMemoryPlugin:
    name = "playbook_memory"

    def __init__(self, store: PlaybookStore | None = None) -> None:
        self._store = store

    def render(self, ctx: PluginObservationContext, cfg: HarnessPluginConfig) -> str | None:
        if not cfg.playbook_memory:
            return None
        store = self._store or PlaybookStore.load(version=cfg.playbook_version)
        rules = store.select_rules(ctx.task_instruction, is_bimanual=ctx.is_bimanual, max_rules=5)
        if not rules:
            return None
        body = "\n".join(f"- {r}" for r in rules)
        return f"## Operational Playbook (`{store.version}`, Zero-Privilege Lessons)\n{body}"


class CrossEmbodimentPriorsPlugin:
    name = "cross_embodiment_priors"

    def render(self, ctx: PluginObservationContext, cfg: HarnessPluginConfig) -> str | None:
        if not cfg.cross_embodiment_priors or not ctx.in_context_demos:
            return None
        lines = [
            f"- Demo {idx} (`{d.get('source_dataset')}` on `{d.get('source_embodiment')}`): "
            f"task=\"{d.get('task_instruction')}\" -> semantic trace `{d.get('semantic_trace_preview')}`"
            for idx, d in enumerate(ctx.in_context_demos[:2], 1)
        ]
        return "## Cross-Embodiment Semantic Action Priors\n" + "\n".join(lines)


class PluginPipeline:
    """Composes toggleable Perceive-Reason-Act plugins and enforces strict zero-privilege safety."""

    def __init__(
        self,
        cfg: HarnessPluginConfig | None = None,
        plugins: list[HarnessPlugin] | None = None,
        playbook_store: PlaybookStore | None = None,
    ) -> None:
        self.cfg = cfg or HarnessPluginConfig()
        self.plugins: list[HarnessPlugin] = plugins or [
            MultiViewGuidancePlugin(),
            ProprioceptionPlugin(),
            AdaptiveStepPlugin(),
            SubtaskPlanningPlugin(),
            ActionHistoryPlugin(),
            PlaybookMemoryPlugin(store=playbook_store),
            CrossEmbodimentPriorsPlugin(),
        ]

    def build_prompt(self, ctx: PluginObservationContext) -> str:
        header = (
            f"You are a Frontier VLM directly operating {ctx.embodiment_summary} through the EAISys URDF Embodied Harness "
            "(**STRICT ZERO-PRIVILEGE MODE**).\n"
            "You have ZERO privileged scene state (no object coordinates, no table height, no object dimensions, no holding oracle). "
            "Infer all spatial relationships visually from the raw RGB camera views alongside raw robot hardware encoders."
        )
        if ctx.extra_embodiment_note:
            header += "\n" + ctx.extra_embodiment_note

        vocab = (
            "## Semantic Action Vocabulary\n"
            "Movement units step the end-effector TCP in each robot arm's own `base_link` frame:\n"
            "- Translation: `MV_FWD` (+X), `MV_BACK` (-X), `MV_LEFT` (+Y), `MV_RIGHT` (-Y), `MV_UP` (+Z), `MV_DOWN` (-Z)\n"
            "- Step gears: `STEP_COARSE` (`0.04 m`), `STEP_FINE` (`0.02 m`), `STEP_ULTRA_FINE` (`0.01 m`)\n"
            "- Rotation: `ROT_ROLL_CW/CCW`, `ROT_PITCH_CW/CCW`, `ROT_YAW_CW/CCW` (`0.262 rad`)\n"
            "- Gripper & control: `GRASP` (close jaws), `RELEASE` (open jaws), `HOLD` (no-op), `DONE` (declare task finished)\n"
            f"Allowed units: {', '.join(ctx.allowed_units)}"
        )

        sections: list[str] = [
            header,
            f"## Natural Language Task Instruction\n{ctx.task_instruction}",
            f"## Interaction Turn\nTurn {ctx.turn_idx} / {ctx.max_turns}",
            vocab,
        ]

        for plugin in self.plugins:
            block = plugin.render(ctx, self.cfg)
            if block:
                sections.append(block)

        if ctx.is_bimanual:
            sections.append(
                "Respond strictly with a JSON object matching (`thought`, `current_subtask`, `left_chunk`, `right_chunk` of 1 to 5 units each)."
            )
        else:
            max_u = 6 if self.cfg.action_chunking else 1
            sections.append(
                f"Respond strictly with a JSON object matching (`thought`, `current_subtask`, `action_chunk` of 1 to {max_u} units)."
            )

        prompt = "\n\n".join(sections)
        assert_zero_privilege_prompt(prompt)
        return prompt


def build_vlm_step_context(
    env: Any,
    turn_idx: int,
    max_turns: int,
    last_subtask: str = "",
    in_context_demos: list[dict[str, Any]] | None = None,
    cfg: HarnessPluginConfig | None = None,
) -> str:
    """Backward-compatible wrapper building a strictly zero-privilege prompt for `TabletopTaskEnv`."""
    interp = env.interpreter
    units = valid_units_for_embodiment(supports_mobile_base=interp.supports_mobile_base)
    arms: list[ArmProprioPacket] = []
    for arm in interp.arm_names:
        tcp = interp.fk_tcp(env.state, arm=arm).p
        q_rad = [round(float(v), 4) for v in env.state.arm_q[arm]]
        g_cmd = "OPEN" if env.state.gripper_open.get(arm, 1.0) > 0.5 else "CLOSED"
        jaw_enc = float(env.jaw_encoder_m.get(arm, 0.0))
        arms.append(
            ArmProprioPacket(
                arm_label=arm,
                robot_id=interp.robot_id,
                step_m=float(env.state.step_m),
                joint_q_rad=q_rad,
                tcp_base_xyz_m=[float(c) for c in tcp],
                gripper_cmd=g_cmd,
                gripper_encoder_qpos=[round(jaw_enc, 4)],
                has_gripper=True,
            )
        )

    ctx = PluginObservationContext(
        embodiment_summary=f"robot `{interp.robot_id}`",
        task_instruction=env.task_instruction,
        turn_idx=turn_idx,
        max_turns=max_turns,
        allowed_units=units,
        camera_guides=[
            (
                "CAM 1 (Left — Top-Down)",
                "up in frame = `+X` (`MV_FWD`), down in frame = `-X` (`MV_BACK`), left in frame = `+Y` (`MV_LEFT`), right in frame = `-Y` (`MV_RIGHT`).",
            ),
            (
                "CAM 2 (Right — Side)",
                "right in frame = `+X` (`MV_FWD`), left in frame = `-X` (`MV_BACK`), up in frame = `+Z` (`MV_UP`), down in frame = `-Z` (`MV_DOWN`).",
            ),
        ],
        arms=arms,
        last_subtask=last_subtask,
        recent_history=list(env.recent_units),
        in_context_demos=list(in_context_demos or ()),
        is_bimanual=False,
    )
    return PluginPipeline(cfg=cfg).build_prompt(ctx)
