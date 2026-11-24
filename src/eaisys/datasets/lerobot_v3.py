"""Adapter for packed LeRobot v3 recordings (e.g., real Dual Franka 16D)."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from eaisys.datasets.schema import ArmTrajectoryStream, RawEpisode, normalize_signal_01


def load_dual_franka_v3_episodes(
    root_dir: str | Path,
    max_episodes: int = 12,
    per_task_limit: int = 4,
) -> list[RawEpisode]:
    """Load packed LeRobot v3 Dual Franka episodes (16D: [left_7, left_g, right_7, right_g])."""
    import pyarrow.parquet as pq

    root = Path(root_dir)
    task_dirs = (
        [root]
        if (root / "meta" / "info.json").is_file()
        else sorted(d for d in root.iterdir() if d.is_dir() and (d / "meta" / "info.json").is_file())
    )
    out: list[RawEpisode] = []

    for tdir in task_dirs:
        task_map: dict[int, str] = {}
        tasks_pq = tdir / "meta" / "tasks.parquet"
        if tasks_pq.is_file():
            tt = pq.read_table(tasks_pq)
            cols = tt.column_names
            for i in range(tt.num_rows):
                idx = int(tt["task_index"][i].as_py()) if "task_index" in cols else i
                txt_col = "task" if "task" in cols else cols[0]
                task_map[idx] = str(tt[txt_col][i].as_py())

        # Accumulate split episodes across multiple chunk parquet files in the same task
        ep_chunks: dict[int, list[np.ndarray]] = {}
        ep_task_idx: dict[int, int] = {}
        pfiles = sorted((tdir / "data").glob("chunk-*/*.parquet"))
        for pf in pfiles:
            tbl = pq.read_table(pf, columns=["observation.state", "episode_index", "task_index"])
            states = np.asarray(tbl["observation.state"].to_pylist(), dtype=np.float64)
            ep_ids = np.asarray(tbl["episode_index"].to_pylist(), dtype=np.int64)
            task_ids = np.asarray(tbl["task_index"].to_pylist(), dtype=np.int64)
            for eid in np.unique(ep_ids):
                mask = ep_ids == eid
                ep_chunks.setdefault(int(eid), []).append(states[mask])
                ep_task_idx.setdefault(int(eid), int(task_ids[mask][0]))
            if len(ep_chunks) > per_task_limit + 1:
                break

        count_task = 0
        for eid in sorted(ep_chunks.keys()):
            s_ep = np.concatenate(ep_chunks[eid], axis=0)
            if len(s_ep) < 10:
                continue
            t_idx = ep_task_idx[eid]
            instr = task_map.get(t_idx) or tdir.name.replace("_", " ")

            ql = s_ep[:, 0:7]
            gl = normalize_signal_01(s_ep[:, 7], min_span=0.01)
            qr = s_ep[:, 8:15]
            gr = normalize_signal_01(s_ep[:, 15], min_span=0.01)

            out.append(
                RawEpisode(
                    dataset_name="dual_franka_real",
                    embodiment_id="dual_franka",
                    episode_id=f"{tdir.name}/ep_{int(eid):04d}",
                    task_instruction=instr,
                    fps=30.0,
                    arms={
                        "left": ArmTrajectoryStream("left", qpos=ql, gripper_open=gl),
                        "right": ArmTrajectoryStream("right", qpos=qr, gripper_open=gr),
                    },
                    metadata={"task_dir": tdir.name},
                )
            )
            count_task += 1
            if len(out) >= max_episodes:
                return out
            if count_task >= per_task_limit:
                break
    return out
