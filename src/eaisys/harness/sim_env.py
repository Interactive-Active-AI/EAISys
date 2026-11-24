"""Multi-view tabletop environment & Pillow image renderer for closed-loop VLM control."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageDraw

from eaisys.harness.interpreter import EmbodimentInterpreter, InterpreterState, StepGroundResult
from eaisys.types import Transform


@dataclass
class SceneObject:
    name: str
    xyz: np.ndarray  # (3,) float64
    size_xyz: tuple[float, float, float] = (0.05, 0.05, 0.06)
    color_rgb: tuple[int, int, int] = (202, 111, 111)
    kind: str = "object"  # "object" | "receptacle"
    held_by_arm: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "xyz": [round(float(c), 4) for c in self.xyz],
            "size_xyz": list(self.size_xyz),
            "kind": self.kind,
            "held_by_arm": self.held_by_arm,
        }


@dataclass
class TabletopTaskEnv:
    """Lightweight physical tabletop simulator driven strictly through an `EmbodimentInterpreter`."""

    interpreter: EmbodimentInterpreter
    task_instruction: str
    objects: dict[str, SceneObject]
    target_object: str
    target_receptacle: str
    table_z: float = 0.16
    grasp_radius_m: float = 0.048
    place_radius_xy_m: float = 0.065
    state: InterpreterState = field(init=False)
    camera_center_xy: np.ndarray = field(init=False)
    jaw_encoder_m: dict[str, float] = field(init=False)
    step_count: int = 0
    recent_units: list[str] = field(default_factory=list)
    grasp_failures: int = 0

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> InterpreterState:
        self.state = self.interpreter.reset()
        arm0 = self.interpreter.arm_names[0]
        self.camera_center_xy = self.interpreter.fk_tcp(self.state, arm=arm0).p[:2].copy()
        self.jaw_encoder_m = {a: round(float(self.interpreter.max_gripper_width_m), 4) for a in self.interpreter.arm_names}
        self.step_count = 0
        self.recent_units.clear()
        self.grasp_failures = 0
        for obj in self.objects.values():
            obj.held_by_arm = None
        return self.state

    def step_unit(self, unit: str, arm: str = "main") -> StepGroundResult:
        prev_state = self.state.clone()
        prev_open = self.state.gripper_open.get(arm, 1.0)
        self.state, res = self.interpreter.apply_unit(unit, self.state, arm=arm)
        self.step_count += 1
        self.recent_units.append(f"{arm}:{unit}" if len(self.interpreter.arm_names) > 1 else unit)

        tcp = self.interpreter.fk_tcp(self.state, arm=arm).p
        # Rigid physical tabletop stop (prevents jaws from passing below solid table surface)
        if unit == "MV_DOWN" and float(tcp[2]) < self.table_z + 0.006:
            self.state = prev_state
            tcp = self.interpreter.fk_tcp(self.state, arm=arm).p
        now_open = self.state.gripper_open.get(arm, 1.0)

        # Handle grasp engagement when gripper transitions open -> closed
        if prev_open > 0.5 and now_open <= 0.5:
            grasped_any = False
            for obj in self.objects.values():
                if obj.kind != "object" or obj.held_by_arm is not None:
                    continue
                dist = float(np.linalg.norm(tcp - obj.xyz))
                min_w = min(obj.size_xyz[0], obj.size_xyz[1])
                if dist <= self.grasp_radius_m and min_w <= self.interpreter.max_gripper_width_m + 1e-3:
                    obj.held_by_arm = arm
                    obj.xyz = tcp.copy()
                    self.jaw_encoder_m[arm] = round(float(min_w), 4)
                    grasped_any = True
                    break
            if not grasped_any:
                self.grasp_failures += 1
                self.jaw_encoder_m[arm] = 0.0

        # Carry attached object with TCP
        for obj in self.objects.values():
            if obj.held_by_arm == arm:
                obj.xyz = tcp.copy()

        # Handle object release when gripper transitions closed -> open
        if prev_open <= 0.5 and now_open > 0.5:
            self.jaw_encoder_m[arm] = round(float(self.interpreter.max_gripper_width_m), 4)
            for obj in self.objects.values():
                if obj.held_by_arm == arm:
                    obj.held_by_arm = None
                    rec = self.objects.get(self.target_receptacle)
                    if rec is not None and float(np.linalg.norm(obj.xyz[:2] - rec.xyz[:2])) <= self.place_radius_xy_m:
                        obj.xyz[2] = float(rec.xyz[2]) + 0.5 * obj.size_xyz[2]
                    else:
                        obj.xyz[2] = max(self.table_z + 0.5 * obj.size_xyz[2], float(self.table_z))
        return res

    def check_success(self) -> tuple[bool, dict[str, Any]]:
        obj = self.objects[self.target_object]
        rec = self.objects[self.target_receptacle]
        xy_dist = float(np.linalg.norm(obj.xyz[:2] - rec.xyz[:2]))
        released = obj.held_by_arm is None
        on_height = abs(float(obj.xyz[2] - rec.xyz[2])) <= 0.08
        ok = bool(released and xy_dist <= self.place_radius_xy_m and on_height)
        return ok, {
            "xy_dist_to_goal_m": round(xy_dist, 4),
            "object_released": released,
            "object_xyz": [round(float(c), 4) for c in obj.xyz],
            "goal_xyz": [round(float(c), 4) for c in rec.xyz],
        }

    def render_multiview_png(self, out_path: str | Path) -> Path:
        """Render a clean zero-privilege 2-panel RGB view (zero rulers, zero gridlines, zero coordinate/object labels)."""
        W, H = 760, 360
        img = Image.new("RGB", (W, H), (251, 249, 247))
        draw = ImageDraw.Draw(img)

        p1 = (16, 36, 366, 340)  # Top-down XY
        p2 = (394, 36, 744, 340)  # Side XZ
        draw.rectangle(p1, fill=(242, 239, 234), outline=(210, 218, 228), width=2)
        draw.rectangle(p2, fill=(244, 246, 249), outline=(210, 218, 228), width=2)
        draw.text((22, 12), "CAM 1: TOP-DOWN", fill=(53, 61, 67))
        draw.text((400, 12), "CAM 2: SIDE", fill=(53, 61, 67))

        # Fixed camera viewport anchored purely at initial robot home XY (zero object or ruler grid dependency)
        cx, cy = float(self.camera_center_xy[0]), float(self.camera_center_xy[1])
        x_min, x_max = cx - 0.18, cx + 0.18
        y_min, y_max = cy - 0.18, cy + 0.18
        z_min, z_max = self.table_z - 0.04, self.table_z + 0.24

        def to_topdown(xyz: np.ndarray) -> tuple[int, int]:
            u = (y_max - float(xyz[1])) / max(1e-4, y_max - y_min)
            v = (x_max - float(xyz[0])) / max(1e-4, x_max - x_min)
            px = int(p1[0] + 18 + np.clip(u, 0.0, 1.0) * (p1[2] - p1[0] - 36))
            py = int(p1[1] + 18 + np.clip(v, 0.0, 1.0) * (p1[3] - p1[1] - 36))
            return px, py

        def to_side(xyz: np.ndarray) -> tuple[int, int]:
            u = (float(xyz[0]) - x_min) / max(1e-4, x_max - x_min)
            v = (z_max - float(xyz[2])) / max(1e-4, z_max - z_min)
            px = int(p2[0] + 18 + np.clip(u, 0.0, 1.0) * (p2[2] - p2[0] - 36))
            py = int(p2[1] + 18 + np.clip(v, 0.0, 1.0) * (p2[3] - p2[1] - 36))
            return px, py

        # Visual table slab in side view (no numeric height label!)
        _, ty_side = to_side(np.array([cx, cy, self.table_z]))
        draw.rectangle((p2[0] + 2, ty_side, p2[2] - 2, p2[3] - 2), fill=(226, 219, 209))
        draw.line((p2[0] + 2, ty_side, p2[2] - 2, ty_side), fill=(180, 170, 158), width=2)

        # Draw receptacles first, then manipulable objects (pure shapes & colors, no text labels!)
        ordered = sorted(self.objects.values(), key=lambda o: 0 if o.kind == "receptacle" else 1)
        for obj in ordered:
            tx, ty = to_topdown(obj.xyz)
            sx, sy = to_side(obj.xyz)
            if obj.kind == "receptacle":
                r = 24
                draw.ellipse((tx - r, ty - r, tx + r, ty + r), fill=obj.color_rgb, outline=(82, 107, 133), width=2)
                draw.rectangle((sx - 24, sy - 4, sx + 24, sy + 4), fill=obj.color_rgb, outline=(82, 107, 133), width=2)
            else:
                r = 13
                draw.rectangle((tx - r, ty - r, tx + r, ty + r), fill=obj.color_rgb, outline=(95, 45, 52), width=2)
                draw.rectangle((sx - 12, sy - 12, sx + 12, sy + 12), fill=obj.color_rgb, outline=(95, 45, 52), width=2)

        # Draw physical gripper jaws (no text overlays!)
        arm_colors = [(65, 95, 138), (116, 99, 140)]
        for idx, arm in enumerate(self.interpreter.arm_names):
            tcp = self.interpreter.fk_tcp(self.state, arm=arm).p
            g_open = self.state.gripper_open.get(arm, 1.0)
            col = arm_colors[idx % len(arm_colors)]
            tx, ty = to_topdown(tcp)
            sx, sy = to_side(tcp)
            jaw = 14 if g_open > 0.5 else 6
            draw.line((tx - jaw, ty - 8, tx - jaw, ty + 8), fill=col, width=3)
            draw.line((tx + jaw, ty - 8, tx + jaw, ty + 8), fill=col, width=3)
            draw.ellipse((tx - 3, ty - 3, tx + 3, ty + 3), fill=col)
            draw.line((sx, p2[1] + 6, sx, sy - 10), fill=(150, 162, 178), width=4)
            draw.line((sx - jaw, sy - 10, sx - jaw, sy + 10), fill=col, width=3)
            draw.line((sx + jaw, sy - 10, sx + jaw, sy + 10), fill=col, width=3)
            draw.line((sx - jaw, sy - 10, sx + jaw, sy - 10), fill=col, width=2)

        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p, format="PNG")
        return p


def make_standard_pick_place_env(
    interpreter: EmbodimentInterpreter,
    seed: int = 0,
) -> TabletopTaskEnv:
    """Create a canonical red-block -> blue-plate tabletop task with continuous unaligned positions."""
    rng = np.random.default_rng(seed)
    s0 = interpreter.reset()
    first_arm = interpreter.arm_names[0]
    tcp0 = interpreter.fk_tcp(s0, arm=first_arm).p
    ws_lo, ws_hi = interpreter.workspace_bounds

    table_z = round(max(float(ws_lo[2]) + 0.015, float(tcp0[2]) - 0.105), 4)
    dx_b = float(rng.uniform(0.035, 0.072))
    dy_b = float(rng.uniform(0.048, 0.088))
    dx_p = float(rng.uniform(-0.038, 0.028))
    dy_p = float(rng.uniform(-0.088, -0.048))

    block_xyz = np.array(
        [
            np.clip(float(tcp0[0]) + dx_b, float(ws_lo[0]) + 0.06, float(ws_hi[0]) - 0.06),
            np.clip(float(tcp0[1]) + dy_b, float(ws_lo[1]) + 0.06, float(ws_hi[1]) - 0.06),
            table_z + 0.02,
        ],
        dtype=np.float64,
    )
    plate_xyz = np.array(
        [
            np.clip(float(tcp0[0]) + dx_p, float(ws_lo[0]) + 0.06, float(ws_hi[0]) - 0.06),
            np.clip(float(tcp0[1]) + dy_p, float(ws_lo[1]) + 0.06, float(ws_hi[1]) - 0.06),
            table_z + 0.005,
        ],
        dtype=np.float64,
    )

    objs = {
        "red_block": SceneObject(
            name="red_block",
            xyz=block_xyz,
            size_xyz=(0.04, 0.04, 0.04),
            color_rgb=(208, 92, 98),
            kind="object",
        ),
        "blue_plate": SceneObject(
            name="blue_plate",
            xyz=plate_xyz,
            size_xyz=(0.11, 0.11, 0.01),
            color_rgb=(172, 198, 224),
            kind="receptacle",
        ),
    }
    return TabletopTaskEnv(
        interpreter=interpreter,
        task_instruction="Pick up the red_block from the table and place it onto the blue_plate, then emit DONE.",
        objects=objs,
        target_object="red_block",
        target_receptacle="blue_plate",
        table_z=table_z,
    )
