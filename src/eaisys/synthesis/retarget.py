"""Retarget multi-embodiment trajectories into shared Semantic Action Space traces."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import numpy as np

from eaisys.datasets.schema import RawEpisode
from eaisys.harness.action_space import delta_xyz_to_semantic_units
from eaisys.urdf import default_robots_root, forward_kinematics_tree, parse_robot_file


@dataclass
class SemanticPriorBank:
    total_episodes: int
    dataset_counts: dict[str, int]
    embodiment_counts: dict[str, int]
    unit_frequencies: dict[str, int]
    exemplars: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_episodes": self.total_episodes,
            "dataset_counts": self.dataset_counts,
            "embodiment_counts": self.embodiment_counts,
            "unit_frequencies": self.unit_frequencies,
            "exemplars": self.exemplars,
        }

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> SemanticPriorBank:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            total_episodes=int(d.get("total_episodes", 0)),
            dataset_counts=dict(d.get("dataset_counts", {})),
            embodiment_counts=dict(d.get("embodiment_counts", {})),
            unit_frequencies=dict(d.get("unit_frequencies", {})),
            exemplars=list(d.get("exemplars", [])),
        )

    def retrieve(self, query: str, top_k: int = 2) -> list[dict[str, Any]]:
        stop = {"the", "a", "an", "and", "to", "on", "in", "into", "of", "from", "with"}
        q_tok = set(query.lower().replace("_", " ").split()) - stop
        scored = []
        for ex in self.exemplars:
            t_tok = set(str(ex.get("task_instruction", "")).lower().replace("_", " ").split()) - stop
            scored.append((len(q_tok & t_tok), ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:top_k]]


def episode_to_semantic_trace(
    episode: RawEpisode,
    robots_root: str | Path | None = None,
    step_m: float = 0.02,
    keyframe_stride: int = 10,
) -> list[str]:
    """Convert a raw robot trajectory + its URDF into discrete semantic action units in `A`."""
    if not episode.arms and episode.gt_skill_spans:
        seq: list[str] = []
        for sp in episode.gt_skill_spans:
            sk = sp.skill.lower()
            if sk in ("pick", "grasp", "grab", "retrieve"):
                seq.extend(["STEP_COARSE", "MV_FWD", "STEP_FINE", "MV_DOWN", "GRASP", "MV_UP"])
            elif sk in ("place", "put"):
                seq.extend(["MV_RIGHT", "MV_DOWN", "RELEASE", "MV_UP"])
            elif sk in ("open", "pull"):
                seq.extend(["MV_FWD", "GRASP", "MV_BACK", "MV_BACK", "RELEASE"])
            elif sk == "insert":
                seq.extend(["STEP_ULTRA_FINE", "MV_DOWN", "MV_DOWN", "RELEASE"])
            else:
                seq.extend(["MV_FWD", "MV_FWD"])
        seq.append("DONE")
        return seq

    rroot = Path(robots_root) if robots_root else default_robots_root()
    urdf_file = rroot / episode.embodiment_id / "robot.urdf"
    tree = parse_robot_file(urdf_file) if urdf_file.is_file() else None

    # Pick the most active arm stream
    best_arm = next(iter(episode.arms.keys()))
    best_pts: np.ndarray | None = None
    best_path_len = -1.0

    for cname, stream in episode.arms.items():
        pts: np.ndarray | None = None
        if stream.eef_pose7_xyzw is not None:
            pts = stream.eef_pose7_xyzw[:, :3].copy()
        elif stream.qpos is not None and tree is not None:
            act_joints = [
                j.name
                for j in tree.joints.values()
                if j.is_actuated and (len(episode.arms) == 1 or cname in j.name.lower())
            ][: stream.qpos.shape[1]]
            leaf = [lk for lk in tree.leaf_links() if len(episode.arms) == 1 or cname in lk.lower()]
            target_lk = leaf[0] if leaf else tree.root_link
            pts = np.zeros((len(stream.qpos), 3), dtype=np.float64)
            for i in range(len(stream.qpos)):
                jmap = {jn: float(v) for jn, v in zip(act_joints, stream.qpos[i])}
                pts[i] = forward_kinematics_tree(tree, jmap)[target_lk].p
        if pts is not None and len(pts) >= 2:
            plen = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
            if plen > best_path_len:
                best_path_len = plen
                best_arm = cname
                best_pts = pts

    if best_pts is None or len(best_pts) < 2:
        return ["HOLD", "DONE"]

    grip = episode.arms[best_arm].gripper_open
    units: list[str] = []
    prev_open = True if grip is None else bool(grip[0] >= 0.5)
    idxs = list(range(0, len(best_pts), max(1, keyframe_stride)))
    if idxs[-1] != len(best_pts) - 1:
        idxs.append(len(best_pts) - 1)

    for k in range(1, len(idxs)):
        i0, i1 = idxs[k - 1], idxs[k]
        d_xyz = best_pts[i1] - best_pts[i0]
        units.extend(delta_xyz_to_semantic_units(d_xyz, step_m=step_m, max_units_per_axis=3))
        if grip is not None:
            seg_g = grip[i0 : i1 + 1]
            now_open = bool(seg_g[-1] >= 0.5)
            if prev_open and (np.min(seg_g) < 0.5):
                units.append("GRASP")
                prev_open = False
            if not prev_open and now_open:
                units.append("RELEASE")
                prev_open = True

    if not units:
        units.append("HOLD")
    units.append("DONE")
    return units


def build_semantic_prior_bank(
    episodes: list[RawEpisode],
    max_exemplars: int = 28,
) -> SemanticPriorBank:
    ds_counts: dict[str, int] = {}
    emb_counts: dict[str, int] = {}
    unit_freq: dict[str, int] = {}
    by_ds: dict[str, list[tuple[RawEpisode, list[str]]]] = {}

    for ep in episodes:
        trace = episode_to_semantic_trace(ep)
        ds_counts[ep.dataset_name] = ds_counts.get(ep.dataset_name, 0) + 1
        emb_counts[ep.embodiment_id] = emb_counts.get(ep.embodiment_id, 0) + 1
        for u in trace:
            unit_freq[u] = unit_freq.get(u, 0) + 1
        by_ds.setdefault(ep.dataset_name, []).append((ep, trace))

    exemplars: list[dict[str, Any]] = []
    ds_keys = list(by_ds.keys())
    r_idx = 0
    while len(exemplars) < max_exemplars and ds_keys:
        progressed = False
        for dk in ds_keys:
            items = by_ds[dk]
            if r_idx >= len(items):
                continue
            progressed = True
            ep, tr = items[r_idx]
            preview = " -> ".join(tr[:18]) + (" -> ..." if len(tr) > 18 else "")
            exemplars.append(
                {
                    "task_instruction": ep.task_instruction,
                    "source_dataset": ep.dataset_name,
                    "source_embodiment": ep.embodiment_id,
                    "episode_id": ep.episode_id,
                    "num_units": len(tr),
                    "semantic_trace_preview": preview,
                    "semantic_units": tr[:48],
                }
            )
            if len(exemplars) >= max_exemplars:
                break
        if not progressed:
            break
        r_idx += 1

    return SemanticPriorBank(
        total_episodes=len(episodes),
        dataset_counts=ds_counts,
        embodiment_counts=emb_counts,
        unit_frequencies=dict(sorted(unit_freq.items(), key=lambda x: -x[1])),
        exemplars=exemplars,
    )
