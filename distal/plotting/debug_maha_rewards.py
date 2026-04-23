"""Quick diagnostic: compute maha distances on the value-training dataset
and print distribution stats to understand reward shape.

Mirrors the exact policy/stats/dataset wiring used in train_value.py when
reward_mode == "maha".
"""

import logging
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.import_utils import register_third_party_plugins

from distal.compute_maha_stats import compute_maha_distances
from distal.maha_reward import load_maha_stats
from distal.train_value import RECAPValueTrainingConfig


def percentile_table(values: np.ndarray, label: str) -> None:
    pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    q = np.percentile(values, pcts)
    print(f"\n{label}")
    print(f"  count={values.size}  mean={values.mean():.4f}  std={values.std():.4f}")
    for p, v in zip(pcts, q):
        print(f"  p{p:>3} = {v:.4f}")


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if x.size == 0 or window <= 1:
        return x
    w = min(window, x.size)
    kernel = np.ones(w)
    num = np.convolve(x, kernel, mode="same")
    denom = np.convolve(np.ones_like(x), kernel, mode="same")
    return num / denom


def plot_all_episodes(
    distances: np.ndarray,
    normalized: np.ndarray,
    episode_indices: np.ndarray,
    success: np.ndarray,
    output_path: Path,
    ma_window: int = 50,
) -> None:
    """Overlay all episodes, color-coded by success/failure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    unique = np.unique(episode_indices)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    ax_d, ax_r = axes

    n_succ = n_fail = 0
    for ep in unique:
        mask = episode_indices == ep
        d = distances[mask]
        r = normalized[mask]
        t = np.arange(d.size)
        is_success = bool(success[mask][0])
        color = "tab:green" if is_success else "tab:red"
        if is_success:
            n_succ += 1
        else:
            n_fail += 1
        ax_d.plot(
            t, moving_average(d, ma_window), color=color, linewidth=1.2, alpha=0.7
        )
        ax_r.plot(
            t, moving_average(r, ma_window), color=color, linewidth=1.2, alpha=0.7
        )

    ax_d.set_title(
        f"raw mahalanobis distance  ({len(unique)} episodes, MA window={ma_window})"
    )
    ax_d.set_xlabel("frame")
    ax_d.set_ylabel("d")
    ax_d.grid(alpha=0.3)

    ax_r.set_title("normalized reward")
    ax_r.set_xlabel("frame")
    ax_r.set_ylabel("reward")
    ax_r.set_ylim(-1.05, 0.05)
    ax_r.grid(alpha=0.3)

    legend = [
        Line2D([0], [0], color="tab:green", lw=2, label=f"success (n={n_succ})"),
        Line2D([0], [0], color="tab:red", lw=2, label=f"failure (n={n_fail})"),
    ]
    ax_d.legend(handles=legend, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"saved per-episode plots -> {output_path}")


def ascii_histogram(values: np.ndarray, bins: int = 20, width: int = 50) -> None:
    counts, edges = np.histogram(values, bins=bins)
    peak = counts.max() or 1
    print("\nhistogram (raw distances):")
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(width * c / peak)
        print(f"  [{lo:8.4f}, {hi:8.4f}]  {c:7d}  {bar}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    register_third_party_plugins()

    cfg = RECAPValueTrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_episodes = 50
    episodes = list(range(num_episodes))
    print(
        f"dataset     = {cfg.repo_id} (first {num_episodes} episodes)\n"
        f"base_policy = {cfg.base_policy}\n"
        f"stats_path  = {cfg.maha_stats_path}\n"
        f"device      = {device}"
    )

    dataset = LeRobotDataset(repo_id=cfg.repo_id, episodes=episodes)
    print(f"frames = {len(dataset.hf_dataset)}  episodes = {len(episodes)}")

    mean, cov_inv = load_maha_stats(cfg.maha_stats_path)
    print(f"stats dim = {mean.shape[0]}")

    policy_cfg = PreTrainedConfig.from_pretrained(cfg.base_policy)
    policy_cfg.pretrained_path = Path(cfg.base_policy)
    policy_cfg.device = str(device)
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    assert isinstance(policy, (PI05Policy, SmolVLAPolicy))
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=str(policy_cfg.pretrained_path)
    )

    distances = compute_maha_distances(
        policy=policy,
        preprocessor=preprocessor,
        dataset=dataset,
        gauss_mean=mean,
        gauss_cov_inv=cov_inv,
        batch_size=cfg.maha_embed_batch_size,
        num_workers=cfg.maha_embed_num_workers,
    )
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    d_min, d_max = float(distances.min()), float(distances.max())
    if d_max > d_min:
        normalized = -(distances - d_min) / (d_max - d_min)
    else:
        normalized = np.zeros_like(distances)

    percentile_table(distances, "raw mahalanobis distance")
    percentile_table(normalized, "normalized reward in [-1, 0]")
    ascii_histogram(distances, bins=20)

    # Per-episode mean raw distance — shows whether episodes differ from each
    # other (between-episode variation) vs. within-episode variation.
    ep_indices = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    per_ep_means = []
    for ep in np.unique(ep_indices):
        per_ep_means.append(distances[ep_indices == ep].mean())
    per_ep_means = np.asarray(per_ep_means)
    print(
        f"\nper-episode mean raw distance: "
        f"std={per_ep_means.std():.4f}  "
        f"range=[{per_ep_means.min():.4f}, {per_ep_means.max():.4f}]"
    )

    # Within-episode spread of raw distances
    within_stds = []
    for ep in np.unique(ep_indices):
        within_stds.append(distances[ep_indices == ep].std())
    within_stds = np.asarray(within_stds)
    print(
        f"within-episode std of raw distance: "
        f"mean={within_stds.mean():.4f}  median={np.median(within_stds):.4f}"
    )

    out = Path("outputs/debug_maha_distances.npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, distances)
    print(f"\nsaved raw distances -> {out}")

    success = np.asarray(dataset.hf_dataset["success"], dtype=bool)
    plot_all_episodes(
        distances=distances,
        normalized=normalized,
        episode_indices=ep_indices,
        success=success,
        output_path=Path("outputs/debug_maha_episodes.png"),
    )


if __name__ == "__main__":
    main()
