"""Evaluate Mahalanobis distance as a failure predictor via AUROC.

Loads pre-computed Mahalanobis stats, embeds dataset frames, computes
per-frame distances, aggregates per episode (max), and reports AUROC
against episode success labels.
"""

from dataclasses import dataclass
from pathlib import Path

import draccus
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging
from safetensors.numpy import load_file
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from distal.compute_maha_stats import compute_mahalanobis_np
from distal.embedding import embed_prefix_pooled


@dataclass
class MahaAurocConfig:
    policy_path: str = "reece-omahoney/adv-libero-base"
    dataset_repo_id: str = "reece-omahoney/libero-10"
    maha_stats_repo_id: str = "reece-omahoney/maha-stats-test"
    num_episodes: int = 50
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42


@draccus.wrap()
def main(cfg: MahaAurocConfig):
    init_logging()
    register_third_party_plugins()

    device = get_safe_torch_device(cfg.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load maha stats
    stats_path = hf_hub_download(
        repo_id=cfg.maha_stats_repo_id,
        filename="stats.safetensors",
        repo_type="dataset",
        force_download=True,
    )
    stats = load_file(stats_path)
    gauss_mean = stats["mean"]
    gauss_cov_inv = stats["cov_inv"]
    pca_mean: np.ndarray | None = stats.get("pca_mean", None)
    pca_components: np.ndarray | None = stats.get("pca_components", None)
    print(
        f"Loaded Mahalanobis stats: mean {gauss_mean.shape}, "
        f"cov_inv {gauss_cov_inv.shape}"
        + (f", PCA {pca_components.shape}" if pca_components is not None else "")
    )

    # Load dataset
    dataset = LeRobotDataset(repo_id=cfg.dataset_repo_id)
    episode_index = np.array(dataset.hf_dataset["episode_index"])
    success = np.array(dataset.hf_dataset["success"])

    # Select shuffled subset of episodes
    unique_episodes = np.unique(episode_index)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(unique_episodes)
    selected_episodes = set(unique_episodes[: cfg.num_episodes].tolist())
    print(f"Selected {len(selected_episodes)} episodes out of {len(unique_episodes)}")

    # Get frame indices for selected episodes
    frame_mask = np.isin(episode_index, list(selected_episodes))
    frame_indices = np.where(frame_mask)[0]

    # Load policy
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    policy_cfg.pretrained_path = Path(cfg.policy_path)
    policy_cfg.device = str(device)
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    assert isinstance(policy, (PI05Policy, SmolVLAPolicy))
    policy.eval()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=str(policy_cfg.pretrained_path)
    )

    # Embed and compute distances
    subset = Subset(dataset, frame_indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    all_dists: list[float] = []
    for batch in tqdm(loader, desc="Computing Maha distances"):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        batch = preprocessor(batch)
        with torch.no_grad():
            emb = embed_prefix_pooled(policy, batch)
            emb_np = emb.cpu().numpy()
            if pca_mean is not None and pca_components is not None:
                emb_np = (emb_np - pca_mean) @ pca_components.T
            dists = compute_mahalanobis_np(emb_np, gauss_mean, gauss_cov_inv)
        all_dists.extend(dists.tolist())

    distances = np.array(all_dists)
    selected_episode_index = episode_index[frame_indices]
    selected_success = success[frame_indices]

    # Aggregate per episode using multiple strategies
    ep_dists: dict[int, np.ndarray] = {}
    ep_success = {}
    for ep in selected_episodes:
        mask = selected_episode_index == ep
        ep_dists[ep] = distances[mask]
        ep_success[ep] = bool(selected_success[mask][0])

    episodes = sorted(selected_episodes)
    labels = np.array([not ep_success[ep] for ep in episodes])  # failure = positive

    n_fail = int(labels.sum())
    n_success = len(labels) - n_fail
    print(f"\nEpisodes: {len(labels)} ({n_success} success, {n_fail} failure)")

    if n_fail == 0 or n_success == 0:
        print("Cannot compute AUROC: only one class present.")
        return

    def mean_last_frac(d: np.ndarray, frac: float) -> float:
        n = max(1, int(len(d) * frac))
        return float(d[-n:].mean())

    def weighted_mean(d: np.ndarray) -> float:
        weights = np.linspace(0.0, 1.0, len(d)) + 1e-6
        return float(np.average(d, weights=weights))

    aggregations = {
        "max": lambda d: d.max(),
        "mean": lambda d: d.mean(),
        "p95": lambda d: np.percentile(d, 95),
        "p90": lambda d: np.percentile(d, 90),
        "mean_top10pct": lambda d: np.sort(d)[int(0.9 * len(d)) :].mean(),
        "mean_last50pct": lambda d: mean_last_frac(d, 0.5),
        "mean_last25pct": lambda d: mean_last_frac(d, 0.25),
        "weighted_mean": weighted_mean,
    }
    print()
    for name, agg_fn in aggregations.items():
        scores = np.array([agg_fn(ep_dists[ep]) for ep in episodes])
        succ = scores[~labels]
        fail = scores[labels]
        auroc = roc_auc_score(labels, scores)
        print(
            f"{name:>15s}: success={succ.mean():.2f}±{succ.std():.2f}  "
            f"failure={fail.mean():.2f}±{fail.std():.2f}  AUROC={auroc:.4f}"
        )


if __name__ == "__main__":
    main()
