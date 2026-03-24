"""Compare SAE reconstruction error vs Mahalanobis distance for OOD detection."""

from dataclasses import dataclass
from pathlib import Path

import draccus
import matplotlib.pyplot as plt
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
from lerobot.utils.utils import init_logging, inside_slurm
from safetensors.numpy import load_file
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from distal.compute_maha_stats import compute_mahalanobis_np
from distal.embedding import embed_prefix_pooled, embed_prefix_tokens
from distal.sae import SparseAutoencoder


@dataclass
class CompareOODConfig:
    policy_path: str = "reece-omahoney/adv-libero-base"
    dataset_repo_id: str = "reece-omahoney/libero-10"
    maha_stats_repo_id: str = "reece-omahoney/maha-stats"
    sae_repo_id: str = "reece-omahoney/sae-libero"
    sae_local_path: str | None = None
    output_dir: str = "outputs/compare_ood"
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 4


@draccus.wrap()
def main(cfg: CompareOODConfig):
    init_logging()
    register_third_party_plugins()

    device = get_safe_torch_device(cfg.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load policy + dataset
    dataset = LeRobotDataset(repo_id=cfg.dataset_repo_id)
    print(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    policy_cfg.pretrained_path = Path(cfg.policy_path)
    policy_cfg.device = str(device)

    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    assert isinstance(policy, (PI05Policy, SmolVLAPolicy))
    policy.eval()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=str(policy_cfg.pretrained_path)
    )

    # Load Mahalanobis stats
    stats_file = hf_hub_download(
        cfg.maha_stats_repo_id, "stats.safetensors", repo_type="dataset"
    )
    stats = load_file(stats_file)
    gauss_mean = stats["mean"]
    gauss_cov_inv = stats["cov_inv"]
    print(f"Loaded Maha stats, dim={gauss_mean.shape[0]}")

    # Load SAE
    if cfg.sae_local_path:
        sae = SparseAutoencoder.from_pretrained(cfg.sae_local_path)
    else:
        sae = SparseAutoencoder.from_hub(cfg.sae_repo_id)
    sae = sae.to(device).eval()
    print(
        f"Loaded SAE: input_dim={sae.config.input_dim}, "
        f"feature_dim={sae.config.feature_dim}"
    )

    # Compute scores in single pass
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    all_maha = []
    all_sae_error = []
    all_success = []

    for batch in tqdm(loader, desc="Computing scores", disable=inside_slurm()):
        success = batch["success"].squeeze(-1).numpy()

        batch_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        batch_device = preprocessor(batch_device)

        with torch.no_grad():
            # Mahalanobis on pooled embeddings
            pooled = embed_prefix_pooled(policy, batch_device)
            maha_dists = compute_mahalanobis_np(
                pooled.cpu().numpy(), gauss_mean, gauss_cov_inv
            )

            # SAE reconstruction error on concatenated tokens
            tokens = embed_prefix_tokens(policy, batch_device)
            sae_errors = sae.reconstruction_error(tokens).cpu().numpy()

        all_maha.extend(maha_dists.tolist())
        all_sae_error.extend(sae_errors.tolist())
        all_success.extend(success.tolist())

    maha_scores = np.array(all_maha)
    sae_scores = np.array(all_sae_error)
    success_labels = np.array(all_success, dtype=bool)

    # OOD labels: failure = 1 (OOD), success = 0 (ID)
    ood_labels = (~success_labels).astype(int)

    print(
        f"\nSamples: {len(ood_labels)} total, "
        f"{success_labels.sum()} success (ID), "
        f"{(~success_labels).sum()} failure (OOD)"
    )

    # Check we have both classes
    if len(np.unique(ood_labels)) < 2:
        print("WARNING: Only one class present, cannot compute ROC/AUC.")
        print("Generating histograms only.\n")
        plot_histograms(maha_scores, sae_scores, success_labels, output_dir)
        return

    # Compute AUCs
    maha_auc = roc_auc_score(ood_labels, maha_scores)
    sae_auc = roc_auc_score(ood_labels, sae_scores)

    print(f"\nMahalanobis AUC: {maha_auc:.4f}")
    print(f"SAE Recon Error AUC: {sae_auc:.4f}")

    print(
        f"\nMaha  — ID mean: {maha_scores[success_labels].mean():.4f}, "
        f"OOD mean: {maha_scores[~success_labels].mean():.4f}"
    )
    print(
        f"SAE   — ID mean: {sae_scores[success_labels].mean():.6f}, "
        f"OOD mean: {sae_scores[~success_labels].mean():.6f}"
    )

    # Generate plots
    plot_histograms(maha_scores, sae_scores, success_labels, output_dir)
    plot_roc_curves(ood_labels, maha_scores, sae_scores, maha_auc, sae_auc, output_dir)
    plot_scatter(maha_scores, sae_scores, success_labels, output_dir)

    print(f"\nPlots saved to {output_dir}")


def plot_histograms(
    maha: np.ndarray,
    sae: np.ndarray,
    success: np.ndarray,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, scores, title in [
        (axes[0], maha, "Mahalanobis Distance"),
        (axes[1], sae, "SAE Reconstruction Error"),
    ]:
        ax.hist(
            scores[success],
            bins=50,
            alpha=0.6,
            color="tab:green",
            label="Success (ID)",
            density=True,
        )
        ax.hist(
            scores[~success],
            bins=50,
            alpha=0.6,
            color="tab:red",
            label="Failure (OOD)",
            density=True,
        )
        ax.set_title(title)
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "histograms.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curves(
    ood_labels: np.ndarray,
    maha: np.ndarray,
    sae: np.ndarray,
    maha_auc: float,
    sae_auc: float,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    for scores, auc, label, color in [
        (maha, maha_auc, "Mahalanobis", "tab:blue"),
        (sae, sae_auc, "SAE Recon Error", "tab:orange"),
    ]:
        fpr, tpr, _ = roc_curve(ood_labels, scores)
        ax.plot(fpr, tpr, color=color, label=f"{label} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("OOD Detection ROC Curves")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_scatter(
    maha: np.ndarray,
    sae: np.ndarray,
    success: np.ndarray,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        maha[success],
        sae[success],
        alpha=0.3,
        s=8,
        c="tab:green",
        label="Success (ID)",
    )
    ax.scatter(
        maha[~success],
        sae[~success],
        alpha=0.3,
        s=8,
        c="tab:red",
        label="Failure (OOD)",
    )
    ax.set_xlabel("Mahalanobis Distance")
    ax.set_ylabel("SAE Reconstruction Error")
    ax.set_title("Mahalanobis vs SAE OOD Scores")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "scatter.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
