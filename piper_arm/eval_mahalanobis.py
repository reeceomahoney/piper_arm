"""Evaluate Mahalanobis distance of VLM prefix embeddings during sim rollouts.

Phase 1: Embed the entire training dataset to fit a Gaussian (mean + covariance)
          over the VLM prefix representations. These stats can be cached to disk.

Phase 2: Roll out the policy in LIBERO and at each timestep compute the VLM
          prefix embedding and its Mahalanobis distance from the training
          distribution. Plot the distance over each trajectory.

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
    reg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed the full dataset and return (mean, cov_inv)."""
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

    print("Fitting Gaussian...")
    mean = embeddings.mean(axis=0)
    cov = np.cov(embeddings, rowvar=False)
    cov += reg * np.eye(cov.shape[0])
    cov_inv = np.linalg.inv(cov)

    return mean, cov_inv


# ──────────────────────────────────────────────────────────────────────
# Phase 2: Rollout with Mahalanobis tracking
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def select_action_with_mahalanobis(
    policy: SmolVLAPolicy,
    batch: dict,
    gauss_mean: np.ndarray,
    gauss_cov_inv: np.ndarray,
) -> tuple[torch.Tensor, dict | None]:
    """Like select_action but also returns Mahalanobis distance of the current obs.

    Returns:
        action: Single action tensor for env stepping.
        stats: Dict with mahalanobis distance on forward-pass steps, None on dequeue.
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
        stats = {"mahalanobis": dist.tolist()}

        # Generate action chunk normally
        action_chunk = policy._get_action_chunk(batch)
        policy._queues[ACTION].extend(
            action_chunk.transpose(0, 1)[: policy.config.n_action_steps]
        )

    action = policy._queues[ACTION].popleft()
    return action, stats


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────


def plot_mahalanobis(results: list[dict], output_dir: Path):
    """Plot per-timestep Mahalanobis distance for each episode."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(12, 6))

    for record in results:
        steps = [t["step"] for t in record["timesteps"]]
        dists = [t["mahalanobis"] for t in record["timesteps"]]
        color = "#2ecc71" if record["success"] else "#e74c3c"
        ax.plot(steps, dists, color=color, alpha=0.7, linewidth=1.0)

    handles = [
        Line2D([0], [0], color="#2ecc71", label="Success"),
        Line2D([0], [0], color="#e74c3c", label="Failure"),
    ]
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
        "--reg",
        type=float,
        default=1e-5,
        help="Regularization added to covariance diagonal",
    )
    parser.add_argument(
        "--load-stats",
        type=str,
        default=None,
        help="Path to gauss_stats.npz from a previous run. Skips dataset embedding.",
    )
    args = parser.parse_args()

    # ── Load policy ──
    suite_name = "libero_10"
    env_cfg = LiberoEnvConfig(suite_name)
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = Path(args.policy_path)

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
        print(f"Loaded Gaussian stats, dim={gauss_mean.shape[0]}")
    else:
        gauss_mean, gauss_cov_inv = fit_gaussian_from_dataset(
            policy=policy,
            preprocessor=preprocessor,
            repo_id=args.repo_id,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            reg=args.reg,
        )

    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = Path(f"outputs/eval_mahalanobis/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Gaussian stats for reuse
    np.savez(
        output_dir / "gauss_stats.npz",
        mean=gauss_mean,
        cov_inv=gauss_cov_inv,
    )
    print(f"Saved Gaussian stats to {output_dir / 'gauss_stats.npz'}")

    # ── Phase 2: Rollout ──
    envs = make_env(env_cfg, n_envs=args.n_episodes)

    results = []

    for task_id, vec_env in envs[suite_name].items():
        task_desc = vec_env.call("task_description")[0]
        n_tasks = len(envs[suite_name])
        print(f"\n=== Task {task_id}/{n_tasks}: {task_desc} ===")

        max_steps = vec_env.call("_max_episode_steps")[0]
        seeds = list(range(args.n_episodes))

        observation, info = vec_env.reset(seed=seeds)
        policy.reset()
        successes = [False] * args.n_episodes
        timestep_metrics = [[] for _ in range(args.n_episodes)]
        done = np.array([False] * args.n_episodes)

        for step in range(max_steps):
            if np.all(done):
                break

            observation = preprocess_observation(observation)
            observation = add_envs_task(vec_env, observation)
            observation = env_preprocessor(observation)
            observation = preprocessor(observation)

            with torch.inference_mode():
                action, stats = select_action_with_mahalanobis(
                    policy, observation, gauss_mean, gauss_cov_inv
                )

            if stats is not None:
                for i in range(args.n_episodes):
                    timestep_metrics[i].append(
                        {"step": step, "mahalanobis": stats["mahalanobis"][i]}
                    )

            action = postprocessor(action)

            action_transition = {ACTION: action}
            action_transition = env_postprocessor(action_transition)
            action_np = action_transition[ACTION].to("cpu").numpy()

            observation, reward, terminated, truncated, info = vec_env.step(action_np)

            if "final_info" in info:
                for i, s in enumerate(info["final_info"]["is_success"].tolist()):
                    if s:
                        successes[i] = True

            done = terminated | truncated | done

        for ep in range(args.n_episodes):
            if timestep_metrics[ep]:
                mean_dist = np.mean([m["mahalanobis"] for m in timestep_metrics[ep]])
            else:
                mean_dist = float("nan")

            episode_record = {
                "task_id": task_id,
                "task_description": task_desc,
                "episode": ep,
                "n_steps": step + 1,
                "success": bool(successes[ep]),
                "mean_mahalanobis": float(mean_dist),
                "timesteps": timestep_metrics[ep],
            }
            results.append(episode_record)

            print(
                f"  Episode {ep}: {step + 1} steps, success={successes[ep]}, "
                f"mean_mahalanobis={mean_dist:.4f}"
            )

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
    plot_mahalanobis(results, output_dir)


if __name__ == "__main__":
    main()
