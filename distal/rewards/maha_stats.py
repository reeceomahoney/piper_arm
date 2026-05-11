"""Compute Mahalanobis statistics (mean, cov_inv) from a dataset and save to disk."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import torch
from huggingface_hub import HfApi
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.processor import rename_stats
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging, inside_slurm
from safetensors.numpy import save_file
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


@torch.no_grad()
def embed_siglip_pooled(policy: PI05Policy, batch: dict) -> torch.Tensor:
    """Mean-pool SigLIP vision-tower embeddings across all camera patches.

    Skips the PaliGemma language model entirely — pure vision representation,
    no language conditioning. Output dim = SigLIP hidden size (1152).
    """
    model = policy.model
    images, img_masks = policy._preprocess_images(batch)
    vision_tower = model.paligemma_with_expert.paligemma.model.vision_tower

    all_embs = []
    all_masks = []
    for img, img_mask in zip(images, img_masks, strict=True):
        img_fp32 = img.to(torch.float32) if img.dtype != torch.float32 else img
        feats = vision_tower(img_fp32).last_hidden_state
        bsize, num_patches = feats.shape[:2]
        all_embs.append(feats)
        all_masks.append(img_mask[:, None].expand(bsize, num_patches))

    embs = torch.cat(all_embs, dim=1)
    mask = torch.cat(all_masks, dim=1).unsqueeze(-1).float()
    return (embs.float() * mask).sum(dim=1) / mask.sum(dim=1)


def compute_mahalanobis_np(
    embeddings: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray
) -> np.ndarray:
    """Mahalanobis distance for each row. embeddings: (N, D)."""
    diff = embeddings - mean[None, :]
    left = diff @ cov_inv
    return np.sqrt(np.sum(left * diff, axis=1))


def compute_maha_distances(
    policy: PI05Policy,
    preprocessor: Any,
    dataset: LeRobotDataset | Subset,
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
            emb = embed_siglip_pooled(policy, batch)
            dists = compute_mahalanobis_np(emb.cpu().numpy(), gauss_mean, gauss_cov_inv)
        all_dists.extend(dists.tolist())
    return np.array(all_dists, dtype=np.float64)


def fit_gaussian_from_dataset(
    policy: PI05Policy,
    preprocessor: Any,
    dataset: LeRobotDataset,
    batch_size: int,
    num_workers: int,
    max_frames: int | None = None,
    subsample_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed the full dataset and return (mean, cov_inv)."""
    device = next(policy.parameters()).device

    print(f"Loading dataset: {dataset.repo_id} with {len(dataset)} frames")
    loader_dataset: torch.utils.data.Dataset = dataset
    if max_frames is not None and max_frames < len(dataset):
        rng = np.random.default_rng(subsample_seed)
        indices = rng.choice(len(dataset), size=max_frames, replace=False)
        loader_dataset = Subset(dataset, sorted(indices.tolist()))
        print(f"Subsampled to {len(loader_dataset)} frames (seed={subsample_seed})")
    dataloader = DataLoader(
        loader_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("Embedding dataset...")
    all_embeddings = []
    for batch in tqdm(dataloader, desc="Embedding"):
        batch_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        batch_device = preprocessor(batch_device)
        emb = embed_siglip_pooled(policy, batch_device)
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
    dataset_repo_id: str = "lerobot/libero_10"
    hub_repo_id: str = "reece-omahoney/pi05-libero-10-maha-stats-siglip"
    output_path: str = "outputs/maha/stats.safetensors"
    device: str = "cuda"
    batch_size: int = 256
    num_workers: int = 16
    max_frames: int | None = 100_000
    subsample_seed: int = 0
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
    dataset = LeRobotDataset(repo_id=cfg.dataset_repo_id, vcodec="auto")

    # Load policy
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    policy_cfg.pretrained_path = Path(cfg.policy_path)
    policy_cfg.device = str(device)

    policy = make_policy(
        cfg=policy_cfg, ds_meta=dataset.meta, rename_map=cfg.rename_map
    )
    assert isinstance(policy, PI05Policy)
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
        max_frames=cfg.max_frames,
        subsample_seed=cfg.subsample_seed,
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
