"""Mahalanobis-distance-based per-step rewards for value training.

Given a base dataset (e.g. ``lerobot/libero``) that a policy was trained on,
``compute_maha_stats.py`` computes the mean/inv-covariance of the policy's
mean-pooled image-token embeddings. This module loads those stats and
computes the Mahalanobis distance for every frame in a value-training
dataset, then min-max normalizes the distances into the ``[-1, 0]`` range so
they can be used in place of the fixed ``-1`` per-step reward.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from safetensors.numpy import load_file

from distal.compute_maha_stats import compute_maha_distances


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


def compute_maha_rewards(
    dataset: LeRobotDataset,
    policy_path: str,
    stats_path: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[int, float]:
    """Return ``{absolute frame index -> reward in [-1, 0]}`` for the dataset."""
    mean, cov_inv = load_maha_stats(stats_path)
    logging.info(f"Loaded maha stats from {stats_path} (dim={mean.shape[0]})")

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = Path(policy_path)
    policy_cfg.device = str(device)
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    assert isinstance(policy, (PI05Policy, SmolVLAPolicy))
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=str(policy_cfg.pretrained_path)
    )

    try:
        distances = compute_maha_distances(
            policy=policy,
            preprocessor=preprocessor,
            dataset=dataset,
            gauss_mean=mean,
            gauss_cov_inv=cov_inv,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    finally:
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    p1 = float(np.percentile(distances, 1))
    p99 = float(np.percentile(distances, 99))
    if p99 <= p1:
        logging.warning(
            f"Degenerate maha distances (p1={p1}, p99={p99}); returning zeros."
        )
        normalized = np.zeros_like(distances)
    else:
        clipped = np.clip(distances, p1, p99)
        normalized = -(clipped - p1) / (p99 - p1)
        mean_reward = float(normalized.mean())
        if mean_reward < 0:
            normalized = normalized * (-1.0 / mean_reward)

    logging.info(
        f"Maha rewards: raw d in [{float(distances.min()):.4f}, "
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
