"""Adapters for ALOHA-Mobile HDF5 and AgiBot World frame-level skill annotations."""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from eaisys.datasets.schema import (
    ArmTrajectoryStream,
    GroundTruthSkillSpan,
    RawEpisode,
    normalize_signal_01,
)


def load_aloha_hdf5_episodes(
    root_dir: str | Path,
    max_episodes: int = 10,
) -> list[RawEpisode]:
    """Load real ALOHA / ALOHA-Mobile HDF5 episodes (`observations/qpos` + `action_eef`)."""
    import h5py

    root = Path(root_dir)
    files = [root] if root.is_file() else sorted(root.glob("episode_*.hdf5"))
    out: list[RawEpisode] = []

    for h5 in files[:max_episodes]:
        with h5py.File(h5, "r") as f:
            qpos = np.asarray(f["observations/qpos"][:], dtype=np.float64)
            aeef = (
                np.asarray(f["action_eef"][:], dtype=np.float64)
                if "action_eef" in f
                else None
            )
            base_act = (
                np.asarray(f["base_action"][:], dtype=np.float64)
                if "base_action" in f
                else None
            )

        ql = qpos[:, 0:6]
        gl = normalize_signal_01(qpos[:, 6], min_span=0.01)
        qr = qpos[:, 7:13]
        gr = normalize_signal_01(qpos[:, 13], min_span=0.01)

        el = aeef[:, 0:7] if aeef is not None and aeef.shape[1] >= 16 else None
        er = aeef[:, 8:15] if aeef is not None and aeef.shape[1] >= 16 else None

        base_twist = None
        if base_act is not None:
            base_twist = np.zeros((len(base_act), 6), dtype=np.float64)
            base_twist[:, 0] = base_act[:, 0]
            base_twist[:, 5] = base_act[:, 1]

        out.append(
            RawEpisode(
                dataset_name="aloha_mobile_real",
                embodiment_id="aloha",
                episode_id=h5.stem,
                task_instruction=root.name.replace("_", " ") if root.is_dir() else "pick and place",
                fps=30.0,
                arms={
                    "left": ArmTrajectoryStream("left", qpos=ql, eef_pose7_xyzw=el, gripper_open=gl),
                    "right": ArmTrajectoryStream("right", qpos=qr, eef_pose7_xyzw=er, gripper_open=gr),
                },
                base_twist6=base_twist,
                metadata={"file_name": h5.name},
            )
        )
    return out


def load_agibot_task_info(
    task_info_dir: str | Path,
    max_episodes: int = 50,
) -> list[RawEpisode]:
    """Load AgiBot World human-annotated frame-level atomic skill intervals."""
    root = Path(task_info_dir)
    files = [root] if root.is_file() else sorted(root.glob("task_*.json"))
    out: list[RawEpisode] = []

    for jf in files:
        entries = json.loads(jf.read_text(encoding="utf-8"))
        for item in entries:
            ep_id = str(item.get("episode_id", ""))
            acfg = item.get("label_info", {}).get("action_config", [])
            spans = [
                GroundTruthSkillSpan(
                    start_frame=int(s["start_frame"]),
                    end_frame=int(s["end_frame"]),
                    skill=str(s.get("skill", "Manipulate")),
                    action_text=str(s.get("action_text", "")),
                )
                for s in acfg
                if "start_frame" in s and "end_frame" in s
            ]
            if not spans:
                continue
            full_instr = " -> ".join(s.action_text for s in spans[:4])
            out.append(
                RawEpisode(
                    dataset_name="agibot_world_beta",
                    embodiment_id="agibot_g1",
                    episode_id=f"{jf.stem}/{ep_id}",
                    task_instruction=full_instr,
                    fps=30.0,
                    arms={},
                    gt_skill_spans=spans,
                    metadata={"task_json": jf.name},
                )
            )
            if len(out) >= max_episodes:
                return out
    return out
