"""Adapter for RoboMIND 2.0 HDF5 datasets (Ark stationary bimanual & Ark-mobile)."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from eaisys.datasets.schema import ArmTrajectoryStream, RawEpisode, normalize_signal_01


def _slug_to_instruction(slug: str, zh_desc: str = "") -> str:
    en = slug.replace("_", " ").strip()
    if zh_desc:
        return f"{en} ({zh_desc.strip()})"
    return en


def load_robomind_episode(
    hdf5_path: str | Path,
    embodiment_id: str = "arx_bimanual",
    dataset_name: str = "robomind2_ark",
) -> RawEpisode:
    import h5py

    p = Path(hdf5_path)
    # Path layout: .../<task_slug>/success_episodes/<ep_id>/data/trajectory.hdf5
    ep_id = p.parents[1].name if len(p.parents) >= 2 else p.stem
    task_dir = p.parents[3] if len(p.parents) >= 4 else p.parent
    task_slug = task_dir.name
    zh_file = task_dir / "zh_description.txt"
    zh_text = zh_file.read_text(encoding="utf-8", errors="ignore").strip() if zh_file.exists() else ""

    with h5py.File(p, "r") as f:
        ql = np.asarray(f["puppet/arm_left_position_align/data"][:], dtype=np.float64)
        qr = np.asarray(f["puppet/arm_right_position_align/data"][:], dtype=np.float64)
        el = np.asarray(f["puppet/end_effector_left_pose_align/data"][:], dtype=np.float64)
        er = np.asarray(f["puppet/end_effector_right_pose_align/data"][:], dtype=np.float64)
        gl_raw = np.asarray(f["puppet/end_effector_left_position_align/data"][:], dtype=np.float64)
        gr_raw = np.asarray(f["puppet/end_effector_right_position_align/data"][:], dtype=np.float64)

        ts = np.asarray(f["camera_observations/timestamp"][:], dtype=np.int64)
        if len(ts) >= 2 and ts[-1] > ts[0]:
            # Timestamps are in nanoseconds or microseconds
            dt_raw = float(np.median(np.diff(ts)))
            if dt_raw > 1e6:
                fps = 1e9 / dt_raw
            elif dt_raw > 1e3:
                fps = 1e6 / dt_raw
            else:
                fps = 30.0
        else:
            fps = 30.0

        base_pose = None
        base_twist = None
        if "puppet/chassis_pose_align/data" in f:
            base_pose = np.asarray(f["puppet/chassis_pose_align/data"][:], dtype=np.float64)
        if "puppet/chassis_twist_align/data" in f:
            base_twist = np.asarray(f["puppet/chassis_twist_align/data"][:], dtype=np.float64)

    arms = {
        "left": ArmTrajectoryStream(
            chain_name="left",
            qpos=ql,
            eef_pose7_xyzw=el,
            gripper_open=normalize_signal_01(gl_raw, min_span=2.0),
        ),
        "right": ArmTrajectoryStream(
            chain_name="right",
            qpos=qr,
            eef_pose7_xyzw=er,
            gripper_open=normalize_signal_01(gr_raw, min_span=2.0),
        ),
    }

    return RawEpisode(
        dataset_name=dataset_name,
        embodiment_id=embodiment_id,
        episode_id=f"{task_slug}/{ep_id}",
        task_instruction=_slug_to_instruction(task_slug, zh_text),
        fps=round(float(fps), 2),
        arms=arms,
        base_pose7_xyzw=base_pose,
        base_twist6=base_twist,
        metadata={"task_slug": task_slug},
    )


def iter_robomind_episodes(
    root_dir: str | Path,
    embodiment_id: str = "arx_bimanual",
    dataset_name: str = "robomind2_ark",
    max_episodes: int = 20,
    per_task_limit: int = 2,
) -> list[RawEpisode]:
    root = Path(root_dir)
    out: list[RawEpisode] = []
    if root.is_file() and root.suffix in (".hdf5", ".h5"):
        return [load_robomind_episode(root, embodiment_id=embodiment_id, dataset_name=dataset_name)]

    task_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    for tdir in task_dirs:
        succ_dir = tdir / "success_episodes"
        if not succ_dir.is_dir():
            continue
        count_for_task = 0
        for ep_dir in sorted(succ_dir.iterdir()):
            h5 = ep_dir / "data" / "trajectory.hdf5"
            if not h5.is_file():
                continue
            out.append(
                load_robomind_episode(h5, embodiment_id=embodiment_id, dataset_name=dataset_name)
            )
            count_for_task += 1
            if len(out) >= max_episodes:
                return out
            if count_for_task >= per_task_limit:
                break
    return out
