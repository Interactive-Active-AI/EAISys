"""Adapters for LeRobot v2.1 Parquet datasets (LIBERO, RoboTwin ALOHA, RoboCasa365)."""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from eaisys.datasets.schema import ArmTrajectoryStream, RawEpisode, normalize_signal_01
from eaisys.types import matrix_to_quat_xyzw, rotvec_to_matrix


def _read_tasks_jsonl(lerobot_root: Path) -> dict[int, str]:
    tasks_file = lerobot_root / "meta" / "tasks.jsonl"
    mapping: dict[int, str] = {}
    if tasks_file.is_file():
        for line in tasks_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            mapping[int(row.get("task_index", 0))] = str(row.get("task", ""))
    return mapping


def _read_fps(lerobot_root: Path, default: float = 20.0) -> float:
    info_file = lerobot_root / "meta" / "info.json"
    if info_file.is_file():
        info = json.loads(info_file.read_text(encoding="utf-8"))
        return float(info.get("fps", default))
    return default


def load_libero_lerobot_episodes(
    root_dir: str | Path,
    max_episodes: int = 16,
) -> list[RawEpisode]:
    """Load LIBERO LeRobot v2.1 episodes (state=[eef_pos(3), eef_axisangle(3), gripper(2)])."""
    import pyarrow.parquet as pq

    root = Path(root_dir)
    suite_roots = (
        [root]
        if (root / "meta" / "info.json").is_file()
        else sorted(d for d in root.iterdir() if d.is_dir() and (d / "meta" / "info.json").is_file())
    )
    episodes: list[RawEpisode] = []
    per_suite = max(1, max_episodes // max(1, len(suite_roots)))

    for sroot in suite_roots:
        tasks = _read_tasks_jsonl(sroot)
        fps = _read_fps(sroot, default=20.0)
        pfiles = sorted((sroot / "data").glob("chunk-*/*.parquet"))
        for pf in pfiles[:per_suite]:
            tbl = pq.read_table(pf)
            states = np.asarray(tbl["observation.state"].to_pylist(), dtype=np.float64)
            t_idx = int(tbl["task_index"][0].as_py()) if "task_index" in tbl.column_names else 0
            instr = tasks.get(t_idx, sroot.name)

            T = len(states)
            pose7 = np.zeros((T, 7), dtype=np.float64)
            pose7[:, :3] = states[:, :3]
            for i in range(T):
                R = rotvec_to_matrix(states[i, 3:6])
                pose7[i, 3:7] = matrix_to_quat_xyzw(R)
            # Finger qpos: states[:, 6] in [0, 0.04], states[:, 7] in [-0.04, 0]
            width = states[:, 6] - states[:, 7]
            grip_open = normalize_signal_01(width, min_span=0.008)

            episodes.append(
                RawEpisode(
                    dataset_name="libero_v21",
                    embodiment_id="franka_panda",
                    episode_id=f"{sroot.name}/{pf.stem}",
                    task_instruction=instr,
                    fps=fps,
                    arms={
                        "main": ArmTrajectoryStream(
                            chain_name="main",
                            qpos=None,
                            eef_pose7_xyzw=pose7,
                            gripper_open=grip_open,
                        )
                    },
                    metadata={"suite": sroot.name},
                )
            )
            if len(episodes) >= max_episodes:
                return episodes
    return episodes


def load_robotwin_lerobot_episodes(
    root_dir: str | Path,
    max_episodes: int = 20,
    per_task_limit: int = 1,
) -> list[RawEpisode]:
    """Load RoboTwin ALOHA 14D LeRobot v2.1 episodes ([left_6, left_g, right_6, right_g])."""
    import pyarrow.parquet as pq

    root = Path(root_dir)
    task_roots = (
        [root]
        if (root / "meta" / "info.json").is_file()
        else sorted(d for d in root.iterdir() if d.is_dir() and (d / "meta" / "info.json").is_file())
    )
    episodes: list[RawEpisode] = []
    for troot in task_roots:
        tasks = _read_tasks_jsonl(troot)
        fps = _read_fps(troot, default=15.0)
        pfiles = sorted((troot / "data").glob("chunk-*/*.parquet"))
        for pf in pfiles[:per_task_limit]:
            tbl = pq.read_table(pf)
            states = np.asarray(tbl["observation.state"].to_pylist(), dtype=np.float64)
            t_idx = int(tbl["task_index"][0].as_py()) if "task_index" in tbl.column_names else 0
            instr = tasks.get(t_idx) or troot.name.replace("_", " ")

            ql = states[:, 0:6]
            gl = np.clip(states[:, 6], 0.0, 1.0)
            qr = states[:, 7:13]
            gr = np.clip(states[:, 13], 0.0, 1.0)

            episodes.append(
                RawEpisode(
                    dataset_name="robotwin_clean",
                    embodiment_id="aloha",
                    episode_id=f"{troot.name}/{pf.stem}",
                    task_instruction=instr,
                    fps=fps,
                    arms={
                        "left": ArmTrajectoryStream("left", qpos=ql, gripper_open=gl),
                        "right": ArmTrajectoryStream("right", qpos=qr, gripper_open=gr),
                    },
                    metadata={"task_name": troot.name},
                )
            )
            if len(episodes) >= max_episodes:
                return episodes
    return episodes


def load_robocasa_lerobot_episodes(
    root_dir: str | Path,
    max_episodes: int = 16,
) -> list[RawEpisode]:
    """Load RoboCasa365 Human300 LeRobot episodes (PandaOmron mobile manipulator)."""
    import pyarrow.parquet as pq

    root = Path(root_dir)
    lerobot_dirs = sorted(root.glob("*/*/*/lerobot"))
    if not lerobot_dirs and (root / "meta" / "info.json").is_file():
        lerobot_dirs = [root]

    episodes: list[RawEpisode] = []
    for ldir in lerobot_dirs:
        pfiles = sorted((ldir / "data").glob("chunk-*/*.parquet"))
        if not pfiles:
            continue
        pf = pfiles[0]
        tbl = pq.read_table(pf)
        states = np.asarray(tbl["observation.state"].to_pylist(), dtype=np.float64)
        actions = np.asarray(tbl["action"].to_pylist(), dtype=np.float64)
        fps = _read_fps(ldir, default=20.0)

        task_name = ldir.parents[1].name if len(ldir.parents) >= 2 else ldir.name
        if "annotation.human.task_description" in tbl.column_names:
            raw_desc = tbl["annotation.human.task_description"][0].as_py()
            instr = str(raw_desc[0] if isinstance(raw_desc, list) and raw_desc else raw_desc)
        else:
            tasks = _read_tasks_jsonl(ldir)
            instr = tasks.get(0, task_name)
        if not instr or instr.isdigit():
            instr = task_name

        # RoboCasa state: [base_pos(3), base_quat(4), eef_pos_rel(3), eef_quat_rel(4), gripper_qpos(2)]
        T = len(states)
        pose7 = np.zeros((T, 7), dtype=np.float64)
        if states.shape[1] >= 14:
            pose7[:, :3] = states[:, 7:10]
            pose7[:, 3:7] = states[:, 10:14]
            base_pose = states[:, 0:7]
        else:
            pose7[:, :3] = states[:, :3]
            pose7[:, 6] = 1.0
            base_pose = None

        # Gripper command in RoboCasa 12D EE-first action is channel 6 (-1 open / +1 close)
        if actions.shape[1] >= 7:
            grip_open = normalize_signal_01(-actions[:, 6], min_span=0.1)
        else:
            grip_open = np.ones(T, dtype=np.float64)

        episodes.append(
            RawEpisode(
                dataset_name="robocasa365_human300",
                embodiment_id="panda_omron",
                episode_id=f"{task_name}/{pf.stem}",
                task_instruction=instr,
                fps=fps,
                arms={
                    "main": ArmTrajectoryStream(
                        chain_name="main",
                        eef_pose7_xyzw=pose7,
                        gripper_open=grip_open,
                    )
                },
                base_pose7_xyzw=base_pose,
                metadata={"task_name": task_name},
            )
        )
        if len(episodes) >= max_episodes:
            break
    return episodes
