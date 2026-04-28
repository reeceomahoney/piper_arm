"""Compute Mahalanobis statistics (mean, cov_inv) from a dataset and save to disk."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import draccus
import numpy as np
import torch
from huggingface_hub import HfApi
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import rename_stats
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging, inside_slurm
from safetensors.numpy import save_file
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


@dataclass
class MahaStatsConfig:
    policy_path: str = "lerobot/pi05-libero"
    dataset_repo_id: str = "lerobot/libero_plus"
    hub_repo_id: str = "reece-omahoney/pi05-libero-plus-maha-stats"
    output_path: str = "outputs/maha/stats.safetensors"
    device: str = "cuda"
    batch_size: int = 512
    num_workers: int = 32
    rename_map: dict[str, str] = field(
        default_factory=lambda: {
            "observation.images.front": "observation.images.image",
            "observation.images.wrist": "observation.images.image2",
        }
    )


@draccus.wrap()
def main(cfg: MahaStatsConfig):
    init_logging()
    register_third_party_plugins()

    device = get_safe_torch_device(cfg.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load dataset
    dataset = LeRobotDataset(repo_id=cfg.dataset_repo_id)

    # Load policy
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    policy_cfg.pretrained_path = Path(cfg.policy_path)
    policy_cfg.device = str(device)

    policy = make_policy(
        cfg=policy_cfg, ds_meta=dataset.meta, rename_map=cfg.rename_map
    )
    assert isinstance(policy, (PI05Policy, SmolVLAPolicy))
    policy.eval()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_cfg.pretrained_path),
        dataset_stats=rename_stats(dataset.meta.stats or {}, cfg.rename_map),
        preprocessor_overrides={
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    # Compute stats
    mean, cov_inv = fit_gaussian_from_dataset(
        policy=policy,
        preprocessor=preprocessor,
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # Save locally
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"mean": mean, "cov_inv": cov_inv}, str(output_path))
    print(f"Saved Mahalanobis stats to {output_path}")

    # Push to Hugging Face Hub
    api = HfApi()
    api.create_repo(cfg.hub_repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(output_path),
        path_in_repo="stats.safetensors",
        repo_id=cfg.hub_repo_id,
        repo_type="dataset",
    )
    print(f"Pushed stats to https://huggingface.co/{cfg.hub_repo_id}")


if __name__ == "__main__":
    main()
