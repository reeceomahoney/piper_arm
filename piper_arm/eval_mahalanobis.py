"""Evaluate Mahalanobis distance of VLM prefix embeddings during sim rollouts.

Phase 1: Embed the entire training dataset to fit a Gaussian (mean + covariance)
          over the VLM prefix representations. These stats can be cached to disk.
          The p99 Mahalanobis distance over training data is also computed and stored
          as the intervention threshold.

Phase 2: Roll out the policy in LIBERO and at each timestep compute the VLM
          prefix embedding and its Mahalanobis distance from the training
          distribution. Plot the distance over each trajectory.

          With --intervene: when the current distance exceeds the training p99,
          the action chunk is reduced to a single step so the policy re-evaluates
          at every subsequent timestep until it returns to in-distribution.

Usage:
    # Full run (embed dataset + rollout):
    python -m piper_arm.eval_mahalanobis \
        --policy-path reece-omahoney/smolvla-libero-256 \
        --repo-id reece-omahoney/libero \
        --n-episodes 1

    # Reuse cached stats from a previous run:
    python -m piper_arm.eval_mahalanobis \
        --policy-path reece-omahoney/smolvla-libero-256 \
        --load-stats outputs/eval_mahalanobis/2026-02-17/15-08-29/gauss_stats.npz \
        --n-episodes 1
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────
# Embedding helpers
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def embed_prefix_pooled(policy: SmolVLAPolicy, batch: dict) -> torch.Tensor:
    """Run a batch through the VLM prefix and return mean-pooled embeddings.

    Args:
        batch: Already preprocessed observation dict on device.

    Returns:
        (B, hidden_dim) mean-pooled over valid prefix tokens.
    """
    model = policy.model

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    (prefix_out, _), _ = model.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=False,
        fill_kv_cache=True,
    )

    mask = prefix_pad_masks.unsqueeze(-1).float()
    pooled = (prefix_out.float() * mask).sum(dim=1) / mask.sum(dim=1)
    return pooled


def compute_mahalanobis_np(
    embeddings: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray
) -> np.ndarray:
    """Mahalanobis distance for each row. embeddings: (N, D)."""
    diff = embeddings - mean[None, :]
    left = diff @ cov_inv
    return np.sqrt(np.sum(left * diff, axis=1))


# ──────────────────────────────────────────────────────────────────────
# Phase 1: Fit Gaussian from dataset
# ──────────────────────────────────────────────────────────────────────


def fit_gaussian_from_dataset(
    policy: SmolVLAPolicy,
    preprocessor,
    repo_id: str,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Embed the full dataset and return (mean, cov_inv, p99_threshold)."""
    device = next(policy.parameters()).device

    print(f"Loading dataset: {repo_id}")
    dataset = LeRobotDataset(repo_id=repo_id)
    dataloader = DataLoader(
        dataset,
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
        emb = embed_prefix_pooled(policy, batch_device)
        all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Embedded {embeddings.shape[0]} frames, dim={embeddings.shape[1]}")

    print("Fitting Gaussian (Ledoit-Wolf shrinkage)...")
    lw = LedoitWolf(assume_centered=False)
    lw.fit(embeddings)
    mean = lw.location_
    cov_inv = lw.precision_
    print(f"  Ledoit-Wolf shrinkage coefficient: {lw.shrinkage_:.4f}")

    train_dists = compute_mahalanobis_np(embeddings, mean, cov_inv)
    p99_threshold = float(np.percentile(train_dists, 99))
    print(f"  Training p99 Mahalanobis distance: {p99_threshold:.4f}")

    return mean, cov_inv, p99_threshold


# ──────────────────────────────────────────────────────────────────────
# Phase 2: Rollout with Mahalanobis tracking
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def select_action_with_mahalanobis(
    policy: SmolVLAPolicy,
    batch: dict,
    gauss_mean: np.ndarray,
    gauss_cov_inv: np.ndarray,
    p99_threshold: float | None = None,
    intervene: bool = False,
) -> tuple[torch.Tensor, dict | None]:
    """Like select_action but also returns Mahalanobis distance of the current obs.

    When intervene=True and p99_threshold is set, reduces the action chunk to a
    single step when OOD (distance > p99_threshold), forcing the policy to
    re-evaluate at every subsequent timestep until it returns in-distribution.

    Returns:
        action: Single action tensor for env stepping.
        stats: Dict with mahalanobis distance and intervention flag on forward-pass
               steps, None on dequeue.
    """
    policy.eval()
    batch = policy._prepare_batch(batch)
    policy._queues = populate_queues(policy._queues, batch, exclude_keys=[ACTION])

    stats = None
    if len(policy._queues[ACTION]) == 0:
        # Compute prefix embedding for Mahalanobis distance
        emb = embed_prefix_pooled(policy, batch)  # (1, D)
        emb_np = emb.cpu().numpy()
        dist = compute_mahalanobis_np(emb_np, gauss_mean, gauss_cov_inv)

        ood = bool(
            intervene and p99_threshold is not None and np.mean(dist) > p99_threshold
        )
        stats = {"mahalanobis": dist.tolist(), "intervention": ood}

        # Generate action chunk; if OOD, only use first action so we re-evaluate
        # on the very next step instead of committing to a full chunk.
        action_chunk = policy._get_action_chunk(batch)
        n_steps = 1 if ood else policy.config.n_action_steps
        policy._queues[ACTION].extend(action_chunk.transpose(0, 1)[:n_steps])

    action = policy._queues[ACTION].popleft()
    return action, stats


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────


def plot_mahalanobis(
    results: list[dict], output_dir: Path, p99_threshold: float | None = None
):
    """Plot per-timestep Mahalanobis distance for each episode."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(12, 6))

    all_dists = []
    has_interventions = any(
        any("intervention" in t for t in record["timesteps"]) for record in results
    )
    for record in results:
        steps = [t["step"] for t in record["timesteps"]]
        dists = [t["mahalanobis"] for t in record["timesteps"]]
        all_dists.extend(dists)
        color = "#2ecc71" if record["success"] else "#e74c3c"
        ax.plot(steps, dists, color=color, alpha=0.7, linewidth=1.0)

        if has_interventions:
            iv_steps = [t["step"] for t in record["timesteps"] if t.get("intervention")]
            iv_dists = [
                t["mahalanobis"] for t in record["timesteps"] if t.get("intervention")
            ]
            if iv_steps:
                ax.scatter(
                    iv_steps, iv_dists, color="#e67e22", s=20, zorder=5, linewidths=0
                )

    if p99_threshold is not None:
        ax.axhline(
            p99_threshold,
            color="#9b59b6",
            linestyle="--",
            linewidth=1.5,
            label=f"Training p99 / intervention threshold ({p99_threshold:.2f})",
        )
    elif all_dists:
        p99 = np.percentile(all_dists, 99)
        ax.axhline(
            p99,
            color="#9b59b6",
            linestyle="--",
            linewidth=1.5,
            label=f"Rollout 99th percentile ({p99:.2f})",
        )

    handles = [
        Line2D([0], [0], color="#2ecc71", label="Success"),
        Line2D([0], [0], color="#e74c3c", label="Failure"),
        Line2D(
            [0],
            [0],
            color="#9b59b6",
            linestyle="--",
            label="Training p99 threshold"
            if p99_threshold is not None
            else "Rollout 99th percentile",
        ),
    ]
    if has_interventions:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#e67e22",
                markersize=5,
                label="Intervention",
            )
        )
    ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mahalanobis Distance")
    ax.set_title("Mahalanobis Distance of VLM Prefix Embeddings During Rollout")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = output_dir / "eval_mahalanobis.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Rollout with Mahalanobis distance tracking of VLM embeddings"
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default="reece-omahoney/smolvla-libero-256",
    )
    parser.add_argument("--repo-id", type=str, default="reece-omahoney/libero")
    parser.add_argument("--n-episodes", type=int, default=1, help="Episodes per task")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for dataset embedding"
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--load-stats",
        type=str,
        default=None,
        help="Path to gauss_stats.npz from a previous run. Skips dataset embedding.",
    )
    parser.add_argument(
        "--intervene",
        action="store_true",
        help=(
            "When set, reduce action chunk to 1 step whenever the Mahalanobis "
            "distance exceeds the training p99 threshold, forcing the policy to "
            "re-evaluate every timestep until it returns in-distribution."
        ),
    )
    args = parser.parse_args()

    # ── Load policy ──
    suite_name = "libero_10"
    env_cfg = LiberoEnvConfig(suite_name, fps=10)
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = Path(args.policy_path)

    envs = make_env(env_cfg, n_envs=args.n_episodes)

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=policy_cfg.pretrained_path
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg, policy_cfg
    )

    # ── Phase 1: Get Gaussian stats ──
    if args.load_stats is not None:
        print(f"Loading cached stats from {args.load_stats}")
        data = np.load(args.load_stats)
        gauss_mean = data["mean"]
        gauss_cov_inv = data["cov_inv"]
        gauss_p99 = float(data["p99"]) if "p99" in data else None
        print(f"Loaded Gaussian stats, dim={gauss_mean.shape[0]}")
        if gauss_p99 is not None:
            print(f"  Loaded training p99 threshold: {gauss_p99:.4f}")
        else:
            print("  Warning: no p99 in cached stats; --intervene will be disabled.")
    else:
        gauss_mean, gauss_cov_inv, gauss_p99 = fit_gaussian_from_dataset(
            policy=policy,
            preprocessor=preprocessor,
            repo_id=args.repo_id,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    if args.intervene and gauss_p99 is None:
        print(
            "Warning: --intervene requested but p99 threshold unavailable; disabling."
        )
        args.intervene = False

    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = Path(f"outputs/eval_mahalanobis/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Gaussian stats for reuse
    save_kwargs = dict(mean=gauss_mean, cov_inv=gauss_cov_inv)
    if gauss_p99 is not None:
        save_kwargs["p99"] = np.array(gauss_p99)
    np.savez(output_dir / "gauss_stats.npz", **save_kwargs)
    print(f"Saved Gaussian stats to {output_dir / 'gauss_stats.npz'}")

    # ── Phase 2: Rollout ──
    results = []

    for task_id, vec_env in envs[suite_name].items():
        task_desc = vec_env.call("task_description")[0]
        n_tasks = len(envs[suite_name])
        print(f"\n=== Task {task_id + 1}/{n_tasks}: {task_desc} ===")

        max_steps = vec_env.call("_max_episode_steps")[0]
        seeds = list(range(args.n_episodes))

        observation, info = vec_env.reset(seed=seeds)
        policy.reset()
        successes = [False] * args.n_episodes
        timestep_metrics = [[] for _ in range(args.n_episodes)]
        done = np.array([False] * args.n_episodes)

        step_bar = tqdm(range(max_steps), desc=f"Task {task_id} steps", leave=False)
        for step in step_bar:
            if np.all(done):
                break

            observation = preprocess_observation(observation)
            observation = add_envs_task(vec_env, observation)
            observation = env_preprocessor(observation)
            observation = preprocessor(observation)

            with torch.inference_mode():
                action, stats = select_action_with_mahalanobis(
                    policy,
                    observation,
                    gauss_mean,
                    gauss_cov_inv,
                    p99_threshold=gauss_p99,
                    intervene=args.intervene,
                )

            if stats is not None:
                mean_dist = np.mean(stats["mahalanobis"])
                postfix = {"maha": f"{mean_dist:.2f}"}
                if args.intervene:
                    postfix["ood"] = stats["intervention"]
                step_bar.set_postfix(postfix)
                for i in range(args.n_episodes):
                    if not done[i]:
                        entry = {
                            "step": step,
                            "mahalanobis": stats["mahalanobis"][i],
                        }
                        if args.intervene:
                            entry["intervention"] = stats["intervention"]
                        timestep_metrics[i].append(entry)

            action = postprocessor(action)

            action_transition = {ACTION: action}
            action_transition = env_postprocessor(action_transition)
            action_np = action_transition[ACTION].to("cpu").numpy()

            observation, reward, terminated, truncated, info = vec_env.step(action_np)

            if "final_info" in info:
                for i, s in enumerate(info["final_info"]["is_success"].tolist()):
                    if s and not done[i]:
                        successes[i] = True

            done = terminated | truncated | done

        for ep in range(args.n_episodes):
            if timestep_metrics[ep]:
                mean_dist = np.mean([m["mahalanobis"] for m in timestep_metrics[ep]])
                n_interventions = sum(
                    1 for m in timestep_metrics[ep] if m.get("intervention", False)
                )
            else:
                mean_dist = float("nan")
                n_interventions = 0

            episode_record = {
                "task_id": task_id,
                "task_description": task_desc,
                "episode": ep,
                "n_steps": step + 1,
                "success": bool(successes[ep]),
                "mean_mahalanobis": float(mean_dist),
                "n_interventions": n_interventions,
                "timesteps": timestep_metrics[ep],
            }
            results.append(episode_record)

            msg = (
                f"  Episode {ep}: {step + 1} steps, success={successes[ep]}, "
                f"mean_mahalanobis={mean_dist:.4f}"
            )
            if args.intervene:
                msg += f", interventions={n_interventions}"
            print(msg)

        vec_env.close()

    # Save results
    output_path = output_dir / "eval_mahalanobis_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    successes = [r["success"] for r in results]
    print(f"\nOverall success rate: {sum(successes)}/{len(successes)}")
    print(
        f"Mean Mahalanobis distance: "
        f"{np.mean([r['mean_mahalanobis'] for r in results]):.4f}"
    )

    # Plot
    plot_mahalanobis(results, output_dir, p99_threshold=gauss_p99)


if __name__ == "__main__":
    main()
