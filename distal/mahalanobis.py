"""Mahalanobis distance computation and Gaussian fitting from dataset embeddings."""

from typing import Any, Union

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.utils import inside_slurm
from torch.utils.data import DataLoader
from tqdm import tqdm

from distal.embedding import embed_prefix_pooled


def compute_mahalanobis_np(
    embeddings: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray
) -> np.ndarray:
    """Mahalanobis distance for each row. embeddings: (N, D)."""
    diff = embeddings - mean[None, :]
    left = diff @ cov_inv
    return np.sqrt(np.sum(left * diff, axis=1))


def compute_maha_distances(
    policy: Union[PI05Policy, SmolVLAPolicy],
    preprocessor: Any,
    dataset: LeRobotDataset,
    gauss_mean: np.ndarray,
    gauss_cov_inv: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    """Compute per-frame Mahalanobis distances for all frames in a dataset.

    Returns:
        (N,) float64 array of distances, one per frame.
    """
    device = next(policy.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    all_dists: list[float] = []
    for batch in tqdm(loader, desc="Maha distances", disable=inside_slurm()):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        batch = preprocessor(batch)
        with torch.no_grad():
            emb = embed_prefix_pooled(policy, batch)
            dists = compute_mahalanobis_np(emb.cpu().numpy(), gauss_mean, gauss_cov_inv)
        all_dists.extend(dists.tolist())
    return np.array(all_dists, dtype=np.float64)


def fit_gaussian_from_dataset(
    policy: Union[PI05Policy, SmolVLAPolicy],
    preprocessor: Any,
    dataset: LeRobotDataset,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed the full dataset and return (mean, cov_inv)."""
    device = next(policy.parameters()).device

    print(f"Loading dataset: {dataset.repo_id} with {len(dataset)} frames")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("Embedding dataset...")
    all_embeddings = []
    for batch in tqdm(dataloader, desc="Embedding", disable=inside_slurm()):
        batch_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        batch_device = preprocessor(batch_device)
        emb = embed_prefix_pooled(policy, batch_device)
        all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Embedded {embeddings.shape[0]} frames, dim={embeddings.shape[1]}")

    print("Fitting Gaussian...")
    mean: np.ndarray = embeddings.mean(axis=0)
    cov_inv: np.ndarray = np.linalg.inv(np.cov(embeddings.T))

    return mean, cov_inv
