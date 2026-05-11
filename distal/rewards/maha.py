"""Mahalanobis-distance-based per-step rewards for value training.

Given a base dataset (e.g. ``lerobot/libero``) that a policy was trained on,
``rewards/maha_stats.py`` computes the mean/inv-covariance of the policy's
mean-pooled image-token embeddings. This module loads those stats and
computes the Mahalanobis distance for every frame in a value-training
dataset, then min-max normalizes the distances into the ``[-1, 0]`` range so
they can be used in place of the fixed ``-1`` per-step reward.
"""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_ASSETS_CACHE
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from safetensors.numpy import load_file, save_file
from torch.utils.data import Subset

from distal.rewards.maha_stats import compute_maha_distances

REWARDS_CACHE_DIR = Path(HF_ASSETS_CACHE) / "distal" / "rewards"


def load_maha_stats(stats_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Resolve ``stats_path`` as a local file or HF dataset repo id."""
    local = Path(stats_path)
    if local.is_file():
        resolved = str(local)
    else:
        resolved = hf_hub_download(
            repo_id=stats_path,
            filename="stats.safetensors",
            repo_type="dataset",
        )
    tensors = load_file(resolved)
    return tensors["mean"], tensors["cov_inv"]


def normalize_distances_to_rewards(
    distances: np.ndarray, dataset: LeRobotDataset, label: str
) -> dict[int, float]:
    """Clip distances to [p1, p99], map to [-1, 0], rescale to mean -1.

    Returns ``{absolute frame index -> reward}`` keyed by the dataset's
    ``index`` column.
    """
    p1 = float(np.percentile(distances, 1))
    p99 = float(np.percentile(distances, 99))
    if p99 <= p1:
        logging.warning(
            f"Degenerate {label} distances (p1={p1}, p99={p99}); returning zeros."
        )
        normalized = np.zeros_like(distances)
    else:
        clipped = np.clip(distances, p1, p99)
        normalized = -(clipped - p1) / (p99 - p1)
        mean_reward = float(normalized.mean())
        if mean_reward < 0:
            normalized = normalized * (-1.0 / mean_reward)

    logging.info(
        f"{label} rewards: raw d in [{float(distances.min()):.4f}, "
        f"{float(distances.max()):.4f}], clip [p1={p1:.4f}, p99={p99:.4f}] -> "
        f"normalized in [{normalized.min():.4f}, {normalized.max():.4f}] "
        f"(mean={normalized.mean():.4f})"
    )

    n = len(dataset.hf_dataset)
    if n != len(normalized):
        raise ValueError(
            f"Distance array length {len(normalized)} does not match "
            f"dataset length {n}."
        )
    rewards: dict[int, float] = {}
    for rel_idx in range(n):
        abs_index = int(dataset.hf_dataset[rel_idx]["index"])
        rewards[abs_index] = float(normalized[rel_idx])
    return rewards


def compute_maha_distances_for_dataset(
    dataset: LeRobotDataset,
    policy_path: str,
    stats_path: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    *,
    frame_indices: list[int] | None = None,
) -> np.ndarray:
    """Return raw per-frame Mahalanobis distances for the dataset.

    If ``frame_indices`` is provided, only those frames are embedded (e.g. for
    AUROC over a sampled subset). The full ``dataset`` is still used to build
    the policy/preprocessor (which need ``dataset.meta``).
    """
    mean, cov_inv = load_maha_stats(stats_path)
    logging.info(f"Loaded maha stats from {stats_path} (dim={mean.shape[0]})")

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = Path(policy_path)
    policy_cfg.device = str(device)
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    assert isinstance(policy, PI05Policy)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=str(policy_cfg.pretrained_path)
    )

    loader_ds: LeRobotDataset | Subset = (
        Subset(dataset, frame_indices) if frame_indices is not None else dataset
    )

    try:
        return compute_maha_distances(
            policy=policy,
            preprocessor=preprocessor,
            dataset=loader_ds,
            gauss_mean=mean,
            gauss_cov_inv=cov_inv,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    finally:
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def dataset_frame_indices(dataset: LeRobotDataset) -> list[int]:
    return [int(dataset.hf_dataset[i]["index"]) for i in range(len(dataset.hf_dataset))]


def rewards_cache_path(sig_dict: dict) -> Path:
    """Content-addressed local path for cached per-frame rewards.

    ``sig_dict`` should include every parameter that affects the rewards
    (mode, dataset repo, policy, stats / kNN hyperparams, demo set, etc.).
    """
    sig = hashlib.sha256(json.dumps(sig_dict, sort_keys=True).encode()).hexdigest()[:16]
    return REWARDS_CACHE_DIR / f"{sig}.safetensors"


def try_load_local_rewards(cache_path: Path) -> dict[int, float] | None:
    if not cache_path.is_file():
        return None
    tensors = load_file(str(cache_path))
    indices = tensors["indices"]
    rewards = tensors["rewards"]
    return {int(i): float(r) for i, r in zip(indices, rewards)}


def save_local_rewards(cache_path: Path, rewards: dict[int, float]) -> None:
    indices_arr = np.array(sorted(rewards.keys()), dtype=np.int64)
    rewards_arr = np.array([rewards[int(i)] for i in indices_arr], dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"indices": indices_arr, "rewards": rewards_arr}, str(cache_path))
    logging.info(f"Saved rewards cache to {cache_path} ({len(rewards)} frames)")


def load_or_compute_rewards(
    dataset: LeRobotDataset,
    sig_dict: dict,
    compute_fn,
    label: str,
    use_cache: bool,
) -> dict[int, float]:
    """Generic wrapper: read content-addressed local cache, recompute on miss."""
    cache_path = rewards_cache_path(sig_dict)
    needed_indices = dataset_frame_indices(dataset)
    needed_set = set(needed_indices)

    cached = try_load_local_rewards(cache_path) if use_cache else None
    if cached is not None:
        missing = needed_set - cached.keys()
        if not missing:
            logging.info(
                f"Loaded cached {label} rewards from {cache_path} "
                f"({len(needed_indices)} frames)"
            )
            return {idx: cached[idx] for idx in needed_indices}
        logging.info(
            f"Cached {label} rewards at {cache_path} are missing "
            f"{len(missing)} of {len(needed_indices)} required frames; recomputing."
        )
    else:
        logging.info(
            f"No cached {label} rewards at {cache_path}; computing "
            f"({len(needed_indices)} frames)."
        )

    rewards = compute_fn()

    if use_cache:
        try:
            save_local_rewards(cache_path, rewards)
        except Exception as error:  # noqa: BLE001
            logging.warning(f"Failed to save {label} rewards cache: {error}")

    return rewards
