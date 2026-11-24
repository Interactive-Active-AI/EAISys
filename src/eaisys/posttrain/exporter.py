"""URDF-conditioned SFT / LoRA / GRPO post-training dataset exporter for Robot-Use Agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from eaisys.sanitize import sanitize_payload, sanitize_text
from eaisys.urdf.parser import parse_urdf_xml


def build_urdf_kinematic_card(urdf_xml: str) -> str:
    """Distill a compact, token-efficient URDF kinematic summary for VLM post-training conditioning."""
    if not urdf_xml.strip():
        return "URDF summary unavailable"
    try:
        root = ET.fromstring(urdf_xml)
        tree = parse_urdf_xml(root)
    except Exception:
        return "URDF summary unavailable"
    actuated = [j for j in tree.joints.values() if j.is_actuated]
    rev_n = sum(1 for j in actuated if j.joint_type == "revolute")
    cont_n = sum(1 for j in actuated if j.joint_type == "continuous")
    pris_n = sum(1 for j in actuated if j.joint_type == "prismatic")
    jnames = ", ".join(j.name for j in actuated[:14])
    return (
        f"robot={tree.name} | root={tree.root_link} | links={len(tree.links)} | "
        f"actuated_dof={len(actuated)} (revolute={rev_n}, continuous={cont_n}, prismatic={pris_n}) | "
        f"joints=[{jnames}]"
    )


def export_posttraining_dataset(
    live_runs_dir: str | Path,
    output_jsonl: str | Path,
    *,
    fmt: str = "llamafactory",
    include_urdf_card: bool = True,
    only_successful: bool = True,
) -> dict[str, Any]:
    """Export zero-privilege 3D MuJoCo agent traces into URDF-conditioned SFT / GRPO JSONL.

    Supported formats:
      - `llamafactory`: ShareGPT-style multimodal conversations (`messages` + `images`) for Qwen-VL / LLaVA LoRA SFT.
      - `openai`: OpenAI chat fine-tuning JSONL (`messages` with text + image_path metadata).
      - `grpo`: RLVR / GRPO trajectory bundles with URDF kinematic card, step traces, and binary episode reward.
    """
    ldir = Path(live_runs_dir)
    out_p = Path(output_jsonl)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fmt_l = fmt.strip().lower()
    if fmt_l not in ("llamafactory", "openai", "grpo"):
        raise ValueError(f"Unsupported export format '{fmt}'. Choose from: llamafactory, openai, grpo.")

    records: list[dict[str, Any]] = []
    episodes_used = 0
    task_counts: dict[str, int] = {}

    for jf in sorted(ldir.glob("unseen_mujoco_*.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        if only_successful and not data.get("success", False):
            continue
        if not data.get("zero_privilege", False):
            continue

        s1 = data.get("stage1_urdf_and_synthesis", {})
        urdf_card = ""
        if include_urdf_card:
            if "urdf_xml" in s1:
                urdf_card = build_urdf_kinematic_card(str(s1["urdf_xml"]))
            elif "urdf_xml_by_arm" in s1 and isinstance(s1["urdf_xml_by_arm"], dict):
                cards = [f"{k}: {build_urdf_kinematic_card(str(v))}" for k, v in s1["urdf_xml_by_arm"].items()]
                urdf_card = " || ".join(cards)

        turns = data.get("stage2_vlm_closed_loop_turns", [])
        if not turns:
            continue

        episodes_used += 1
        tname = str(data.get("task_name") or "").strip()
        if not tname:
            env_str = str(data.get("environment", ""))
            if "Stack" in env_str or "stack" in str(data.get("run_id", "")):
                tname = "Stack"
            else:
                tname = "unknown"
        task_counts[tname] = task_counts.get(tname, 0) + 1

        if fmt_l == "grpo":
            records.append(
                sanitize_payload(
                    {
                        "episode_id": data.get("run_id", jf.stem),
                        "task_name": tname,
                        "robot_id": data.get("robot_id", ""),
                        "is_bimanual": bool(data.get("is_bimanual", False)),
                        "urdf_kinematic_card": urdf_card,
                        "task_instruction": data.get("task_instruction", ""),
                        "reward": 1.0 if data.get("success") else 0.0,
                        "zero_privilege": True,
                        "turns": [
                            {
                                "turn": t.get("turn"),
                                "image": f"website/{t.get('observation_image', '')}",
                                "prompt": t.get("prompt", ""),
                                "response": t.get("raw_response", ""),
                                "action_chunk": t.get("action_chunk"),
                                "left_chunk": t.get("left_chunk"),
                                "right_chunk": t.get("right_chunk"),
                            }
                            for t in turns
                        ],
                    }
                )
            )
            continue

        for t in turns:
            user_text = str(t.get("prompt", ""))
            if urdf_card:
                user_text = f"## URDF Kinematic Contract\n{urdf_card}\n\n{user_text}"
            assistant_text = str(t.get("raw_response", ""))
            img_rel = f"website/{t.get('observation_image', '')}"
            if fmt_l == "llamafactory":
                rec = {
                    "messages": [
                        {"role": "user", "content": f"<image>\n{sanitize_text(user_text)}"},
                        {"role": "assistant", "content": sanitize_text(assistant_text)},
                    ],
                    "images": [img_rel],
                    "metadata": {
                        "run_id": data.get("run_id", jf.stem),
                        "task_name": tname,
                        "robot_id": data.get("robot_id", ""),
                        "turn": t.get("turn"),
                        "zero_privilege": True,
                    },
                }
            else:
                rec = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a zero-privilege URDF-conditioned Robot-Use Agent outputting structured semantic action chunks.",
                        },
                        {"role": "user", "content": sanitize_text(user_text)},
                        {"role": "assistant", "content": sanitize_text(assistant_text)},
                    ],
                    "image_path": img_rel,
                    "metadata": {
                        "run_id": data.get("run_id", jf.stem),
                        "task_name": tname,
                        "robot_id": data.get("robot_id", ""),
                        "turn": t.get("turn"),
                    },
                }
            records.append(sanitize_payload(rec))

    with out_p.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "format": fmt_l,
        "output_jsonl": str(out_p),
        "episodes_exported": episodes_used,
        "samples_exported": len(records),
        "task_counts": task_counts,
        "include_urdf_card": include_urdf_card,
    }
