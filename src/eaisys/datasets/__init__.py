"""Dataset ingestion dispatcher across supported multi-embodiment formats."""

from __future__ import annotations

from pathlib import Path

from eaisys.datasets.aloha_agibot import load_agibot_task_info, load_aloha_hdf5_episodes
from eaisys.datasets.lerobot_v2 import (
    load_libero_lerobot_episodes,
    load_robocasa_lerobot_episodes,
    load_robotwin_lerobot_episodes,
)
from eaisys.datasets.lerobot_v3 import load_dual_franka_v3_episodes
from eaisys.datasets.robomind import iter_robomind_episodes, load_robomind_episode
from eaisys.datasets.schema import ArmTrajectoryStream, GroundTruthSkillSpan, RawEpisode


def load_episodes_by_adapter(
    adapter: str,
    path: str | Path,
    robot_id: str = "",
    dataset_name: str = "",
    max_episodes: int = 16,
) -> list[RawEpisode]:
    if adapter == "robomind_hdf5":
        return iter_robomind_episodes(
            path,
            embodiment_id=robot_id or "arx_bimanual",
            dataset_name=dataset_name or "robomind2",
            max_episodes=max_episodes,
        )
    if adapter == "lerobot_v2_libero":
        return load_libero_lerobot_episodes(path, max_episodes=max_episodes)
    if adapter == "lerobot_v2_bimanual":
        return load_robotwin_lerobot_episodes(path, max_episodes=max_episodes)
    if adapter == "lerobot_v2_robocasa":
        return load_robocasa_lerobot_episodes(path, max_episodes=max_episodes)
    if adapter == "lerobot_v3_packed":
        return load_dual_franka_v3_episodes(path, max_episodes=max_episodes)
    if adapter == "aloha_hdf5":
        return load_aloha_hdf5_episodes(path, max_episodes=max_episodes)
    if adapter == "agibot_task_info":
        return load_agibot_task_info(path, max_episodes=max_episodes)
    raise ValueError(f"Unknown dataset adapter '{adapter}'.")


__all__ = [
    "ArmTrajectoryStream",
    "GroundTruthSkillSpan",
    "RawEpisode",
    "iter_robomind_episodes",
    "load_agibot_task_info",
    "load_aloha_hdf5_episodes",
    "load_dual_franka_v3_episodes",
    "load_episodes_by_adapter",
    "load_libero_lerobot_episodes",
    "load_robocasa_lerobot_episodes",
    "load_robomind_episode",
    "load_robotwin_lerobot_episodes",
]
