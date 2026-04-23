"""Plot sample ground-truth normalized return curves built from the
train_value config.

Mirrors the reward/return construction in ``distal/train_value.py``
(``_build_frame_targets``): per-step rewards come from either a fixed -1
or the Mahalanobis reward pipeline, a terminal reward of 0 (success) or
``-c_fail`` (failure) is appended, and the reverse-cumulative sum is
normalized by the per-task max episode length and clipped to ``[-1, 0]``.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.import_utils import register_third_party_plugins

from distal.train_value import (
    RECAPValueTrainingConfig,
    _build_frame_targets,
    _load_episode_success_from_dataset,
)


def select_sample_episode_ids(
    success_by_episode: dict[int, int],
    num_success: int,
    num_failure: int,
) -> list[int]:
    success_ids = sorted(ep for ep, s in success_by_episode.items() if s == 1)
    failure_ids = sorted(ep for ep, s in success_by_episode.items() if s == 0)
    return success_ids[:num_success] + failure_ids[:num_failure]


def plot_return_curves(
    episode_curves: dict[int, tuple[np.ndarray, np.ndarray | None, int, str]],
    output_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    has_steps = any(steps is not None for _, steps, _, _ in episode_curves.values())

    n_succ = n_fail = 0
    for ep_idx, (returns, steps_returns, success, task) in sorted(
        episode_curves.items()
    ):
        color = "tab:green" if success == 1 else "tab:red"
        n_succ += int(success == 1)
        n_fail += int(success == 0)
        t = np.arange(returns.size)
        ax.plot(t, returns, color=color, linewidth=1.4, alpha=0.8, label=f"ep {ep_idx}")
        ax.annotate(
            f"ep {ep_idx}",
            xy=(t[-1], returns[-1]),
            xytext=(3, 0),
            textcoords="offset points",
            fontsize=7,
            color=color,
        )
        if steps_returns is not None:
            ax.plot(
                np.arange(steps_returns.size),
                steps_returns,
                color=color,
                linewidth=1.0,
                alpha=0.6,
                linestyle=":",
            )

    ax.set_ylim(-1.05, 0.05)
    ax.set_xlabel("trajectory step")
    ax.set_ylabel("normalized return")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    legend = [
        Line2D([0], [0], color="tab:green", lw=2, label=f"success (n={n_succ})"),
        Line2D([0], [0], color="tab:red", lw=2, label=f"failure (n={n_fail})"),
    ]
    if has_steps:
        legend.append(
            Line2D(
                [0],
                [0],
                color="black",
                lw=1.2,
                linestyle=":",
                label="steps mode return",
            )
        )
    ax.legend(handles=legend, loc="lower right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    register_third_party_plugins()

    cfg = RECAPValueTrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_success = 5
    num_failure = 5
    num_episodes_to_load = 50

    print(
        f"dataset     = {cfg.repo_id} (first {num_episodes_to_load} episodes)\n"
        f"reward_mode = {cfg.reward_mode}\n"
        f"c_fail      = {cfg.c_fail}\n"
        f"device      = {device}"
    )

    dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
        root=cfg.root,
        revision=cfg.revision,
        episodes=list(range(num_episodes_to_load)),
    )
    success_by_episode = _load_episode_success_from_dataset(dataset)

    step_rewards: dict[int, float] | None = None
    if cfg.reward_mode == "maha":
        from distal.maha_reward import compute_maha_rewards

        print(f"computing maha rewards with {cfg.base_policy} ...")
        step_rewards = compute_maha_rewards(
            dataset=dataset,
            policy_path=cfg.base_policy,
            stats_path=cfg.maha_stats_path,
            device=device,
            batch_size=cfg.maha_embed_batch_size,
            num_workers=cfg.maha_embed_num_workers,
        )

    frame_targets = _build_frame_targets(
        dataset=dataset,
        success_by_episode=success_by_episode,
        c_fail=cfg.c_fail,
        num_value_bins=cfg.num_value_bins,
        step_rewards=step_rewards,
    )
    steps_frame_targets: list | None = None
    if step_rewards is not None:
        steps_frame_targets = _build_frame_targets(
            dataset=dataset,
            success_by_episode=success_by_episode,
            c_fail=cfg.c_fail,
            num_value_bins=cfg.num_value_bins,
            step_rewards=None,
        )

    sample_ids = select_sample_episode_ids(
        success_by_episode=success_by_episode,
        num_success=num_success,
        num_failure=num_failure,
    )
    print(f"selected {len(sample_ids)} episodes: {sample_ids}")

    curves: dict[int, tuple[np.ndarray, np.ndarray | None, int, str]] = {}
    for ep_idx in sample_ids:
        ep_targets = sorted(
            (t for t in frame_targets if t.episode_index == ep_idx),
            key=lambda t: t.frame_index,
        )
        if not ep_targets:
            continue
        returns = np.array([t.target_value for t in ep_targets], dtype=np.float32)

        steps_returns: np.ndarray | None = None
        if steps_frame_targets is not None:
            steps_ep_targets = sorted(
                (t for t in steps_frame_targets if t.episode_index == ep_idx),
                key=lambda t: t.frame_index,
            )
            if steps_ep_targets:
                steps_returns = np.array(
                    [t.target_value for t in steps_ep_targets], dtype=np.float32
                )

        curves[ep_idx] = (
            returns,
            steps_returns,
            ep_targets[0].success,
            ep_targets[0].task,
        )

    output_path = Path("outputs/gt_returns.png")
    plot_return_curves(
        episode_curves=curves,
        output_path=output_path,
        title=(
            f"ground-truth normalized returns  "
            f"(reward_mode={cfg.reward_mode}, c_fail={cfg.c_fail}, "
            f"num_value_bins={cfg.num_value_bins})"
        ),
    )
    print(f"saved -> {output_path}")


if __name__ == "__main__":
    main()
