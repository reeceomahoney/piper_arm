"""Per-step reward source configs used by RECAP value-network training.

Each subclass owns its own knobs and a ``compute_step_rewards`` method that
returns per-frame rewards keyed by absolute frame index (or ``None`` to fall
back to the default fixed -1 per step).
"""

import abc
import logging
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import torch
from huggingface_hub.constants import HF_ASSETS_CACHE
from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class RewardConfig(draccus.ChoiceRegistry, abc.ABC):
    """Per-step reward source for value-network training.

    Subclasses are registered as draccus choices; pick one with
    ``--reward.type=[steps|maha|knn]`` and override its fields.
    """

    # Cache computed per-frame rewards locally (HF_ASSETS_CACHE/distal/rewards),
    # content-addressed by mode + dataset + relevant hyperparams.
    cache: bool = True

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @classmethod
    def default_choice_name(cls) -> str | None:
        return "knn"

    @abc.abstractmethod
    def compute_step_rewards(
        self,
        dataset: LeRobotDataset,
        device: torch.device,
    ) -> dict[int, float] | None:
        """Return per-frame rewards keyed by absolute frame index, or ``None``
        to fall back to the default fixed -1 per step."""


@RewardConfig.register_subclass("steps")
@dataclass
class StepsRewardConfig(RewardConfig):
    """Fixed -1 per step (no embedding-based shaping)."""

    def compute_step_rewards(
        self,
        dataset: LeRobotDataset,
        device: torch.device,
    ) -> None:
        return None


@RewardConfig.register_subclass("maha")
@dataclass
class MahaRewardConfig(RewardConfig):
    """Normalized Mahalanobis distance of each frame's VLM embedding from the
    base training distribution (distal/rewards/maha.py)."""

    base_policy: str = "lerobot/pi05-libero"
    stats_path: str = "reece-omahoney/pi05-maha-stats"
    embed_batch_size: int = 128
    embed_num_workers: int = 8

    def compute_step_rewards(
        self,
        dataset: LeRobotDataset,
        device: torch.device,
    ) -> dict[int, float]:
        from distal.rewards.maha import load_or_compute_maha_rewards

        logging.info(
            f"Loading or computing Mahalanobis-distance rewards using "
            f"{self.base_policy} (dataset cache: {dataset.repo_id})..."
        )
        return load_or_compute_maha_rewards(
            dataset=dataset,
            policy_path=self.base_policy,
            stats_path=self.stats_path,
            device=device,
            batch_size=self.embed_batch_size,
            num_workers=self.embed_num_workers,
            use_cache=self.cache,
        )


@RewardConfig.register_subclass("knn")
@dataclass
class KnnRewardConfig(RewardConfig):
    """Normalized mean kNN distance from each frame's VLM embedding to the
    base demo embeddings (distal/rewards/knn.py). Defaults match
    distal/auroc.py so kNN-AUROC and kNN-reward use the same demo set."""

    base_policy: str = "lerobot/pi05-libero"
    embed_batch_size: int = 128
    embed_num_workers: int = 8
    k: int = 10
    metric: str = "l2"  # "l2" or "cosine"
    chunk_size: int = 4096
    demo_dataset_repo_id: str = "lerobot/libero_10"
    demo_max_frames: int | None = 50_000
    demo_subsample_seed: int = 0
    demo_rename_map: dict[str, str] = field(
        default_factory=lambda: {
            "observation.images.front": "observation.images.image",
            "observation.images.wrist": "observation.images.image2",
        }
    )
    demo_embs_cache_dir: str = str(Path(HF_ASSETS_CACHE) / "distal" / "demo_embs")

    def compute_step_rewards(
        self,
        dataset: LeRobotDataset,
        device: torch.device,
    ) -> dict[int, float]:
        from distal.rewards.knn import load_or_compute_knn_rewards

        logging.info(
            f"Loading or computing kNN-distance rewards using {self.base_policy} "
            f"(demos: {self.demo_dataset_repo_id}, k={self.k}, "
            f"metric={self.metric}, dataset cache: {dataset.repo_id})..."
        )
        return load_or_compute_knn_rewards(
            dataset=dataset,
            policy_path=self.base_policy,
            device=device,
            batch_size=self.embed_batch_size,
            num_workers=self.embed_num_workers,
            knn_k=self.k,
            knn_metric=self.metric,
            knn_chunk_size=self.chunk_size,
            demo_dataset_repo_id=self.demo_dataset_repo_id,
            demo_max_frames=self.demo_max_frames,
            demo_subsample_seed=self.demo_subsample_seed,
            demo_rename_map=self.demo_rename_map,
            demo_embs_cache_dir=self.demo_embs_cache_dir,
            use_cache=self.cache,
        )
