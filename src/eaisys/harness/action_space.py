"""Shared Semantic Action Space A for VLM-centric closed-loop robot control."""

from __future__ import annotations

import math
from typing import Any
import numpy as np


TRANSLATION_UNITS: dict[str, tuple[float, float, float]] = {
    "MV_FWD": (1.0, 0.0, 0.0),
    "MV_BACK": (-1.0, 0.0, 0.0),
    "MV_LEFT": (0.0, 1.0, 0.0),
    "MV_RIGHT": (0.0, -1.0, 0.0),
    "MV_UP": (0.0, 0.0, 1.0),
    "MV_DOWN": (0.0, 0.0, -1.0),
}

ROTATION_UNITS: dict[str, tuple[float, float, float]] = {
    "ROT_ROLL_CW": (1.0, 0.0, 0.0),
    "ROT_ROLL_CCW": (-1.0, 0.0, 0.0),
    "ROT_PITCH_CW": (0.0, 1.0, 0.0),
    "ROT_PITCH_CCW": (0.0, -1.0, 0.0),
    "ROT_YAW_CW": (0.0, 0.0, -1.0),
    "ROT_YAW_CCW": (0.0, 0.0, 1.0),
}

GRIPPER_UNITS = ("GRASP", "RELEASE")

STEP_MODE_UNITS: dict[str, float] = {
    "STEP_COARSE": 0.04,      # 4.0 cm transport step
    "STEP_FINE": 0.02,        # 2.0 cm standard step
    "STEP_ULTRA_FINE": 0.01,  # 1.0 cm precision alignment step
}

MOBILE_BASE_UNITS: dict[str, tuple[float, float, float]] = {
    "BASE_FWD": (0.15, 0.0, 0.0),
    "BASE_BACK": (-0.15, 0.0, 0.0),
    "BASE_LEFT": (0.0, 0.15, 0.0),
    "BASE_RIGHT": (0.0, -0.15, 0.0),
    "BASE_YAW_LEFT": (0.0, 0.0, math.radians(15.0)),
    "BASE_YAW_RIGHT": (0.0, 0.0, -math.radians(15.0)),
}

CONTROL_UNITS = ("HOLD", "DONE")

SINGLE_ARM_UNITS: tuple[str, ...] = (
    tuple(TRANSLATION_UNITS.keys())
    + tuple(ROTATION_UNITS.keys())
    + GRIPPER_UNITS
    + tuple(STEP_MODE_UNITS.keys())
    + CONTROL_UNITS
)

ALL_SEMANTIC_UNITS: tuple[str, ...] = SINGLE_ARM_UNITS + tuple(MOBILE_BASE_UNITS.keys())


def valid_units_for_embodiment(supports_mobile_base: bool = False) -> list[str]:
    units = list(SINGLE_ARM_UNITS)
    if supports_mobile_base:
        units.extend(MOBILE_BASE_UNITS.keys())
    return units


def single_arm_step_json_schema(supports_mobile_base: bool = False) -> dict[str, Any]:
    allowed = valid_units_for_embodiment(supports_mobile_base=supports_mobile_base)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["thought", "current_subtask", "action_chunk"],
        "properties": {
            "thought": {
                "type": "string",
                "description": "Concise visual + proprioceptive spatial reasoning for this turn.",
            },
            "current_subtask": {
                "type": "string",
                "description": "Active subtask description or 'completed'.",
            },
            "action_chunk": {
                "type": "array",
                "minItems": 1,
                "maxItems": 6,
                "items": {"type": "string", "enum": allowed},
                "description": "1 to 6 semantic action units to execute sequentially before re-observing.",
            },
        },
    }


def bimanual_step_json_schema(supports_mobile_base: bool = False) -> dict[str, Any]:
    allowed = valid_units_for_embodiment(supports_mobile_base=supports_mobile_base)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["thought", "current_subtask", "left_chunk", "right_chunk"],
        "properties": {
            "thought": {
                "type": "string",
                "description": "Concise spatial reasoning for coordinated dual-arm control.",
            },
            "current_subtask": {
                "type": "string",
                "description": "Active subtask description or 'completed'.",
            },
            "left_chunk": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {"type": "string", "enum": allowed},
            },
            "right_chunk": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {"type": "string", "enum": allowed},
            },
        },
    }


def delta_xyz_to_semantic_units(
    delta_xyz: np.ndarray,
    step_m: float = 0.02,
    max_units_per_axis: int = 4,
) -> list[str]:
    """Quantize a 3D world-frame displacement into discrete semantic translation units."""
    dx, dy, dz = float(delta_xyz[0]), float(delta_xyz[1]), float(delta_xyz[2])
    units: list[str] = []
    for val, pos_u, neg_u in (
        (dx, "MV_FWD", "MV_BACK"),
        (dy, "MV_LEFT", "MV_RIGHT"),
        (dz, "MV_UP", "MV_DOWN"),
    ):
        n = min(max_units_per_axis, int(round(abs(val) / max(1e-4, step_m))))
        if n > 0:
            units.extend([pos_u if val > 0 else neg_u] * n)
    return units
