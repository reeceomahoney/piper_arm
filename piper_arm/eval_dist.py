"""Evaluate Mahalanobis distance of VLM prefix embeddings during sim rollouts.

Supports both PI05 and SmolVLA policies (auto-detected from checkpoint).

Phase 1: Embed the entire training dataset to fit a Gaussian (mean + covariance)
          over the VLM prefix representations. These stats can be cached to disk.

Phase 2: Roll out the policy in LIBERO and at each timestep compute the VLM
          prefix embedding and its Mahalanobis distance from the training
          distribution. Plot the distance over each trajectory.

          With --intervene: when the moving average of distances exceeds
          INTERVENTION_K times the baseline (mean of the first MA_WINDOW
          forward-pass distances), the action chunk is reduced to a single step
          so the policy re-evaluates at every subsequent timestep.

Usage:
    # Full run (embed dataset + rollout) with PI05:
    python -m piper_arm.eval_dist \
        --policy-path lerobot/pi05_libero_finetuned \
        --repo-id reece-omahoney/libero \
        --n-episodes 1

    # Full run with SmolVLA:
    python -m piper_arm.eval_dist \
        --policy-path lerobot/smolvla_libero_finetuned \
        --repo-id reece-omahoney/libero \
        --n-episodes 1

    # Reuse cached stats from a previous run:
    python -m piper_arm.eval_dist \
        --policy-path lerobot/pi05_libero_finetuned \
        --load-stats outputs/eval_dist/2026-02-17/15-08-29/gauss_stats.npz \
        --n-episodes 1
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, cast

import draccus
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, make_att_2d_masks
from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
)
from lerobot.policies.smolvla.modeling_smolvla import (
    make_att_2d_masks as smolvla_make_att_2d_masks,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.utils.utils import inside_slurm
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────
# Embedding helpers
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def embed_prefix_pooled(
    policy: Union[PI05Policy, SmolVLAPolicy], batch: dict
) -> torch.Tensor:
    """Run a batch through the VLM prefix and return mean-pooled embeddings.

    Supports both PI05 and SmolVLA policies. Only image tokens are included
    in the pooling; language and state tokens are excluded.

    Args:
        batch: Already preprocessed observation dict on device.

    Returns:
        (B, hidden_dim) mean-pooled over image tokens.
    """
    if isinstance(policy, PI05Policy):
        prefix_out, prefix_pad_masks = _embed_prefix_pi05(policy, batch)
    elif isinstance(policy, SmolVLAPolicy):
        prefix_out, prefix_pad_masks = _embed_prefix_smolvla(policy, batch)
    else:
        raise TypeError(f"Unsupported policy type: {type(policy)}")

    mask = prefix_pad_masks.unsqueeze(-1).float()
    pooled = (prefix_out.float() * mask).sum(dim=1) / mask.sum(dim=1)
    return pooled


def _embed_prefix_pi05(policy: PI05Policy, batch: dict):
    """PI05: images + language → PaliGemma prefix forward (4D attention masks).

    Returns prefix_out and a pooling mask covering image tokens only
    (language token positions are masked out).
    """
    model = policy.model
    images, img_masks = policy._preprocess_images(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids: torch.LongTensor = torch.cumsum(prefix_pad_masks, dim=1) - 1  # type: ignore[assignment]

    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
    prefix_att_2d_masks_4d = prefix_att_2d_masks_4d.to(dtype=prefix_embs.dtype)

    (prefix_out, _), _ = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=cast(list[torch.FloatTensor], [prefix_embs, None]),
        use_cache=False,
    )

    # Pooling mask: image tokens only (prefix layout: [img...][lang...])
    n_lang = lang_tokens.shape[1]
    n_img = prefix_embs.shape[1] - n_lang
    vis_mask = prefix_pad_masks.clone()
    vis_mask[:, n_img:] = False

    return prefix_out, vis_mask


def _embed_prefix_smolvla(policy: SmolVLAPolicy, batch: dict):
    """SmolVLA: images + language + state → SmolVLM prefix forward (3D masks).

    Returns prefix_out and a pooling mask covering image tokens only
    (language and state token positions are masked out).
    """
    model = policy.model
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    prefix_att_2d_masks = smolvla_make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids: torch.LongTensor = torch.cumsum(prefix_pad_masks, dim=1) - 1  # type: ignore[assignment]

    (prefix_out, _), _ = model.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=cast(list[torch.FloatTensor], [prefix_embs, None]),
        use_cache=False,
        fill_kv_cache=True,
    )

    # Pooling mask: image tokens only
    # Prefix layout: [img_special+img...][lang...][state...][padding...]
    # State tokens are the only positions with prefix_att_masks == 1
    n_lang = lang_tokens.shape[1]
    state_positions = torch.where(prefix_att_masks[0] == 1)[0]
    first_state = state_positions[0].item()
    lang_start = first_state - n_lang

    img_mask = prefix_pad_masks.clone()
    img_mask[:, lang_start:] = False

    return prefix_out, img_mask


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
    policy: Union[PI05Policy, SmolVLAPolicy],
    preprocessor: Any,
    dataset: str,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed the full dataset and return (mean, cov_inv)."""
    device = next(policy.parameters()).device

    print(f"Loading dataset: {dataset}")
    lerobot_dataset = LeRobotDataset(repo_id=dataset)
    dataloader = DataLoader(
        lerobot_dataset,
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

    print("Fitting Gaussian (Ledoit-Wolf shrinkage)...")
    lw = LedoitWolf(assume_centered=False)
    lw.fit(embeddings)
    mean: np.ndarray = lw.location_
    cov_inv: np.ndarray = lw.precision_  # type: ignore[assignment]
    print(f"  Ledoit-Wolf shrinkage coefficient: {lw.shrinkage_:.4f}")

    return mean, cov_inv


# ──────────────────────────────────────────────────────────────────────
# Phase 2: Rollout with Mahalanobis tracking
# ──────────────────────────────────────────────────────────────────────


MA_WINDOW = 10
INTERVENTION_K = 1.1


def _get_action_queue(policy: Union[PI05Policy, SmolVLAPolicy]):
    """Return the action deque for the given policy type."""
    if isinstance(policy, PI05Policy):
        return policy._action_queue
    elif isinstance(policy, SmolVLAPolicy):
        return policy._queues[ACTION]
    raise TypeError(f"Unsupported policy type: {type(policy)}")


@torch.no_grad()
def select_action_with_mahalanobis(
    policy: Union[PI05Policy, SmolVLAPolicy],
    batch: dict,
    gauss_mean: np.ndarray,
    gauss_cov_inv: np.ndarray,
    dist_histories: list[list[float]],
    done: np.ndarray,
    intervene: bool = False,
) -> tuple[torch.Tensor, dict | None]:
    """Like select_action but also returns Mahalanobis distance of the current obs.

    Supports both PI05 and SmolVLA policies.

    When intervene=True, reduces the action chunk to a single step when the
    per-episode moving average of distances exceeds INTERVENTION_K times the
    baseline (mean of the first MA_WINDOW forward-pass distances),
    forcing the policy to re-evaluate every timestep until it settles back down.

    Args:
        dist_histories: Per-episode list of distances from forward passes so far
                        this episode. Updated in-place. Reset between episodes.

    Returns:
        action: Single action tensor for env stepping.
        stats: Dict with mahalanobis distance and intervention flag on forward-pass
               steps, None on dequeue.
    """
    policy.eval()
    action_queue = _get_action_queue(policy)

    stats = None
    if len(action_queue) == 0:
        # Compute prefix embedding for Mahalanobis distance
        emb = embed_prefix_pooled(policy, batch)  # (1, D)
        emb_np = emb.cpu().numpy()
        dist = compute_mahalanobis_np(emb_np, gauss_mean, gauss_cov_inv)

        # Update per-episode histories (skip done episodes)
        for i, d in enumerate(dist.tolist()):
            if not done[i]:
                dist_histories[i].append(d)

        # Intervene if any episode's moving average has risen k× above its baseline
        ood = False
        if intervene:
            for history in dist_histories:
                if len(history) >= MA_WINDOW:
                    baseline = np.mean(history[:MA_WINDOW])
                    current_ma = np.mean(history[-MA_WINDOW:])
                    if current_ma > INTERVENTION_K * baseline:
                        ood = True
                        break

        stats = {"mahalanobis": dist.tolist(), "intervention": ood}

        # Generate action chunk; if OOD, only use first action so we re-evaluate
        # on the very next step instead of committing to a full chunk.
        action_chunk = policy.predict_action_chunk(batch)
        n_steps = 1 if ood else policy.config.n_action_steps
        action_queue.extend(action_chunk[:, :n_steps].transpose(0, 1))

    action = action_queue.popleft()
    return action, stats


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────


def plot_mahalanobis(results: list[dict], output_dir: Path):
    """Plot per-timestep Mahalanobis distance (moving average) for each episode."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(12, 6))

    has_interventions = any(
        any(t.get("intervention") for t in record["timesteps"]) for record in results
    )
    for record in results:
        steps = np.array([t["step"] for t in record["timesteps"]])
        dists = np.array([t["mahalanobis"] for t in record["timesteps"]])
        color = "#2ecc71" if record["success"] else "#e74c3c"

        if len(dists) >= MA_WINDOW:
            kernel = np.ones(MA_WINDOW) / MA_WINDOW
            ma_dists = np.convolve(dists, kernel, mode="valid")
            ma_steps = steps[MA_WINDOW - 1 :]
        else:
            ma_dists = dists
            ma_steps = steps

        # Find first intervention index in original timestep array
        first_iv_idx = next(
            (i for i, t in enumerate(record["timesteps"]) if t.get("intervention")),
            None,
        )
        if first_iv_idx is not None:
            split = max(0, first_iv_idx - MA_WINDOW + 1)
            ax.plot(
                ma_steps[: split + 1],
                ma_dists[: split + 1],
                color=color,
                alpha=0.7,
                linewidth=1.0,
            )
            ax.plot(
                ma_steps[split:],
                ma_dists[split:],
                color=color,
                alpha=0.7,
                linewidth=1.0,
                linestyle=":",
            )
        else:
            ax.plot(ma_steps, ma_dists, color=color, alpha=0.7, linewidth=1.0)

    handles = [
        Line2D([0], [0], color="#2ecc71", label="Success"),
        Line2D([0], [0], color="#e74c3c", label="Failure"),
    ]
    if has_interventions:
        handles.append(
            Line2D(
                [0],
                [0],
                color="gray",
                linestyle=":",
                linewidth=1.0,
                label="Intervening",
            )
        )
    ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(f"Mahalanobis Distance (moving avg, window={MA_WINDOW})")
    ax.set_title("Mahalanobis Distance of VLM Prefix Embeddings During Rollout")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = output_dir / "eval_dist.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


@dataclass
class EvalMahalanobisConfig:
    policy_path: str = "reece-omahoney/smolvla-libero-16-chunk"
    dataset: str = "reece-omahoney/libero"
    n_episodes: int = 5
    batch_size: int = 32
    num_workers: int = 8
    load_stats: Optional[str] = None
    intervene: bool = False


@draccus.wrap()  # type: ignore[misc]
def main(cfg: EvalMahalanobisConfig):
    # ── Load policy ──
    suite_name = "libero_10"
    env_cfg = LiberoEnvConfig(suite_name, fps=10)
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    policy_cfg.pretrained_path = Path(cfg.policy_path)

    envs = make_env(env_cfg, n_envs=cfg.n_episodes)

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    assert isinstance(policy, (PI05Policy, SmolVLAPolicy))
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=str(policy_cfg.pretrained_path)
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg, policy_cfg
    )

    # ── Phase 1: Get Gaussian stats ──
    if cfg.load_stats is not None:
        print(f"Loading cached stats from {cfg.load_stats}")
        data = np.load(cfg.load_stats)
        gauss_mean = data["mean"]
        gauss_cov_inv = data["cov_inv"]
        print(f"Loaded Gaussian stats, dim={gauss_mean.shape[0]}")
    else:
        gauss_mean, gauss_cov_inv = fit_gaussian_from_dataset(
            policy=policy,
            preprocessor=preprocessor,
            dataset=cfg.dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )

    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = Path(f"outputs/eval_dist/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Gaussian stats for reuse
    np.savez(output_dir / "gauss_stats.npz", mean=gauss_mean, cov_inv=gauss_cov_inv)
    print(f"Saved Gaussian stats to {output_dir / 'gauss_stats.npz'}")

    # ── Phase 2: Rollout ──
    results = []

    for task_id, vec_env in envs[suite_name].items():
        task_desc = vec_env.call("task_description")[0]  # type: ignore[attr-defined]
        n_tasks = len(envs[suite_name])
        print(f"\n=== Task {task_id + 1}/{n_tasks}: {task_desc} ===")

        max_steps = vec_env.call("_max_episode_steps")[0]  # type: ignore[attr-defined]
        seeds = list(range(cfg.n_episodes))

        observation, info = vec_env.reset(seed=seeds)  # type: ignore[arg-type]
        policy.reset()
        successes = [False] * cfg.n_episodes
        timestep_metrics: list[list[dict[str, Any]]] = [
            [] for _ in range(cfg.n_episodes)
        ]
        dist_histories: list[list[float]] = [[] for _ in range(cfg.n_episodes)]
        done = np.array([False] * cfg.n_episodes)
        step = 0

        for step in tqdm(range(max_steps), leave=False, disable=inside_slurm()):
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
                    dist_histories=dist_histories,
                    done=done,
                    intervene=cfg.intervene,
                )

            if stats is not None:
                for i in range(cfg.n_episodes):
                    if not done[i]:
                        entry = {
                            "step": step,
                            "mahalanobis": stats["mahalanobis"][i],
                        }
                        if cfg.intervene:
                            entry["intervention"] = stats["intervention"]
                        timestep_metrics[i].append(entry)

            action = postprocessor(action)

            action_transition = {ACTION: action}
            action_transition = env_postprocessor(action_transition)
            action_np = action_transition[ACTION].to("cpu").numpy()

            observation, _, terminated, truncated, info = vec_env.step(action_np)

            if "final_info" in info:
                for i, s in enumerate(info["final_info"]["is_success"].tolist()):
                    if s and not done[i]:
                        successes[i] = True

            done = terminated | truncated | done

        for ep in range(cfg.n_episodes):
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
            if cfg.intervene:
                msg += f", interventions={n_interventions}"
            print(msg)

        vec_env.close()

    # Save results
    output_path = output_dir / "eval_dist_results.json"
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
    if cfg.intervene:
        # False positives
        total_interventions = sum(r["n_interventions"] for r in results)
        success_interventions = sum(
            r["n_interventions"] for r in results if r["success"]
        )
        print(
            f"Interventions on successful episodes: "
            f"{success_interventions}/{total_interventions} "
            f"({100 * success_interventions / total_interventions:.1f}%)"
        )

        # False negatives
        failed_no_intervention = sum(
            1 for r in results if not r["success"] and r["n_interventions"] == 0
        )
        n_failed = sum(1 for r in results if not r["success"])
        print(
            f"Unsuccessful episodes with no intervention: "
            f"{failed_no_intervention}/{n_failed} "
            f"({100 * failed_no_intervention / n_failed:.1f}%)"
        )

    # Plot
    plot_mahalanobis(results, output_dir)


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
