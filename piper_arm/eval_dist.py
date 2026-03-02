"""Rollout trace capture — run episodes and save camera videos and
Mahalanobis distance traces for offline visualization.

Supports both PI05 and SmolVLA policies (auto-detected from checkpoint).

Phase 1: Embed the entire training dataset to fit a Gaussian (mean + covariance)
          over the VLM prefix representations. These stats can be cached to disk.

Phase 2: Roll out the policy in LIBERO and at each timestep compute the VLM
          prefix embedding and its Mahalanobis distance from the training
          distribution. Outputs per-episode MP4 videos, NPZ distance traces,
          and a summary JSON.

Usage:
    python -m piper_arm.eval_dist \
        --policy-path lerobot/pi05_libero_finetuned \
        --dataset reece-omahoney/libero \
        --n-episodes 1

    # With cached Gaussian stats:
    python -m piper_arm.eval_dist \
        --policy-path lerobot/pi05_libero_finetuned \
        --load-stats outputs/eval_dist/.../gauss_stats.npz \
        --n-episodes 3
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Union, cast

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


# ──────────────────────────────────────────────────────────────────────
# Phase 2: Rollout with trace capture
# ──────────────────────────────────────────────────────────────────────


MA_WINDOW = 10


def _get_action_queue(policy: Union[PI05Policy, SmolVLAPolicy]):
    """Return the action deque for the given policy type."""
    if isinstance(policy, PI05Policy):
        return policy._action_queue
    elif isinstance(policy, SmolVLAPolicy):
        return policy._queues[ACTION]
    raise TypeError(f"Unsupported policy type: {type(policy)}")


def build_frame(
    obs: dict[str, Any], action: torch.Tensor, features: dict[str, dict]
) -> dict[str, Any]:
    """Extract dataset-relevant fields from a preprocessed observation."""
    frame = {"action": action}

    for key in features:
        val = obs.get(key)
        if val is None:
            continue

        if key.startswith("observation.images."):
            # (C, H, W) float [0,1] → (H, W, C) uint8
            frame[key] = (val.permute(1, 2, 0) * 255).to(torch.uint8)
        else:
            frame[key] = val

    return frame


def _run_episode_capture(
    policy: Union[PI05Policy, SmolVLAPolicy],
    vec_env: Any,
    preprocessor: Any,
    postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    gauss_mean: np.ndarray,
    gauss_cov_inv: np.ndarray,
    seed: int,
    desc: str = "",
) -> dict[str, Any]:
    """Run a single episode, capturing full observations, actions, and
    Mahalanobis traces.

    No interventions — the episode runs to completion or truncation.

    Returns:
        Dict with keys: success, n_steps, mean_distance,
        trace_steps, trace_distances,
        actions (list of np.ndarray),
        observations (list of dicts — preprocessed observation per step).
    """
    max_steps = vec_env.call("_max_episode_steps")[0]
    observation, info = vec_env.reset(seed=[seed])
    policy.reset()

    success = False
    done = False
    step = 0
    trace_steps: list[int] = []
    trace_distances: list[float] = []
    dist_history: list[float] = []
    actions: list[np.ndarray] = []
    observations: list[dict[str, torch.Tensor]] = []

    for step in tqdm(
        range(max_steps),
        desc=desc,
        leave=False,
        disable=inside_slurm(),
    ):
        if done:
            break

        observation = preprocess_observation(observation)
        observation = add_envs_task(vec_env, observation)
        observation = env_preprocessor(observation)

        observations.append(
            {
                k: v[0].cpu() if isinstance(v, torch.Tensor) else v
                for k, v in observation.items()
            }
        )

        observation = preprocessor(observation)

        with torch.inference_mode():
            policy.eval()
            action_queue = _get_action_queue(policy)

            if len(action_queue) == 0:
                emb = embed_prefix_pooled(policy, observation)
                emb_np = emb.cpu().numpy()
                dist = compute_mahalanobis_np(emb_np, gauss_mean, gauss_cov_inv)

                dist_val = dist[0].item()
                dist_history.append(dist_val)
                trace_steps.append(step)
                trace_distances.append(dist_val)

                action_chunk = policy.predict_action_chunk(observation)
                n_action_steps = policy.config.n_action_steps
                action_chunk = action_chunk[:, :n_action_steps]
                action_queue.extend(action_chunk.transpose(0, 1))

            action = action_queue.popleft()

        actions.append(action[0].cpu())
        action = postprocessor(action)
        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action_np = action_transition[ACTION].to("cpu").numpy()

        observation, _, terminated, truncated, info = vec_env.step(action_np)

        if "final_info" in info:
            if info["final_info"]["is_success"].item():
                success = True

        done = bool(terminated | truncated)

    mean_dist = float(np.mean(trace_distances)) if trace_distances else float("nan")

    return {
        "success": success,
        "n_steps": step + 1 if max_steps > 0 else 0,
        "mean_distance": mean_dist,
        "trace_steps": trace_steps,
        "trace_distances": trace_distances,
        "actions": actions,
        "observations": observations,
    }


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────


def _plot_traces(results: list[dict], output_dir: Path) -> None:
    """Plot per-episode Mahalanobis distance traces colored by success/failure."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(12, 6))

    for result in results:
        steps = np.array(result["trace_steps"])
        dists = np.array(result["trace_distances"])
        color = "#2ecc71" if result["success"] else "#e74c3c"

        if len(dists) >= MA_WINDOW:
            kernel = np.ones(MA_WINDOW) / MA_WINDOW
            ma_dists = np.convolve(dists, kernel, mode="valid")
            ma_steps = steps[MA_WINDOW - 1 :]
        else:
            ma_dists = dists
            ma_steps = steps

        ax.plot(ma_steps, ma_dists, color=color, alpha=0.7, linewidth=1.0)

    handles = [
        Line2D([0], [0], color="#2ecc71", label="Success"),
        Line2D([0], [0], color="#e74c3c", label="Failure"),
    ]
    ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(f"Mahalanobis Distance (MA, w={MA_WINDOW})")
    ax.set_title("Mahalanobis Distance of VLM Prefix Embeddings")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = output_dir / "eval_dist.png"
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


@dataclass
class EvalDistConfig:
    policy_path: str = "reece-omahoney/smolvla-libero-16-chunk"
    base_dataset_repo_id: str = "reece-omahoney/libero"
    n_episodes: int = 50
    batch_size: int = 32
    num_workers: int = 8
    load_stats: str | None = "outputs/eval_dist/2026-03-02/13-11-13/gauss_stats.npz"
    dataset_repo_id: str | None = "reece-omahoney/rollout-dataset"
    output_dir: str = "outputs/eval_dist"


@draccus.wrap()  # type: ignore[misc]
def main(cfg: EvalDistConfig):
    # ── Load policy ──
    suite_name = "libero_10"
    env_cfg = LiberoEnvConfig(suite_name, fps=10, task_ids=[9])
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    policy_cfg.pretrained_path = Path(cfg.policy_path)

    envs = make_env(env_cfg, n_envs=1)

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    assert isinstance(policy, (PI05Policy, SmolVLAPolicy))
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=str(policy_cfg.pretrained_path)
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg, policy_cfg
    )
    base_dataset = LeRobotDataset(repo_id=cfg.base_dataset_repo_id)

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
            dataset=base_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )

    # ── Output directory ──
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = Path(cfg.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a "latest" symlink for convenience
    latest_link = Path(cfg.output_dir) / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(output_dir.resolve())

    # Save Gaussian stats for reuse
    np.savez(output_dir / "gauss_stats.npz", mean=gauss_mean, cov_inv=gauss_cov_inv)
    print(f"Saved Gaussian stats to {output_dir / 'gauss_stats.npz'}")

    # ── Dataset setup ──
    dataset = None
    if cfg.dataset_repo_id:
        dataset = LeRobotDataset.create(
            repo_id=cfg.dataset_repo_id,
            fps=int(base_dataset.meta.fps),
            features=base_dataset.meta.features,
            root=output_dir / "dataset",
        )

    # ── Phase 2: Rollout with capture ──
    summary_entries: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []

    try:
        for task_id, vec_env in envs[suite_name].items():
            task_desc = vec_env.call("task_description")[0]
            n_tasks = len(envs[suite_name])
            print(f"\n=== Task {task_id + 1}/{n_tasks}: {task_desc} ===")

            for ep in range(cfg.n_episodes):
                result = _run_episode_capture(
                    policy=policy,
                    vec_env=vec_env,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    gauss_mean=gauss_mean,
                    gauss_cov_inv=gauss_cov_inv,
                    seed=ep,
                    desc=f"  Ep {ep}",
                )

                # Build LeRobot dataset incrementally
                if dataset is not None:
                    for frame_idx in range(len(result["observations"])):
                        frame = build_frame(
                            result["observations"][frame_idx],
                            result["actions"][frame_idx],
                            dataset.meta.features,
                        )
                        frame["task"] = task_desc
                        dataset.add_frame(frame)
                    dataset.save_episode()

                status = "OK" if result["success"] else "FAIL"
                print(
                    f"  Episode {ep}: {status} | "
                    f"{result['n_steps']} steps | "
                    f"mean_dist={result['mean_distance']:.2f}"
                )

                summary_entries.append(
                    {
                        "episode": ep,
                        "task_id": task_id,
                        "task_description": task_desc,
                        "success": result["success"],
                        "n_steps": result["n_steps"],
                        "mean_distance": result["mean_distance"],
                    }
                )
                all_results.append(result)

            vec_env.close()
    finally:
        if dataset is not None:
            dataset.finalize()
            print(
                f"Dataset saved to {dataset.root}"
                f" ({dataset.num_episodes} episodes, {dataset.num_frames} frames)"
            )

    print(f"\nOutputs saved to {output_dir}")

    # ── Plot ──
    _plot_traces(all_results, output_dir)


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
