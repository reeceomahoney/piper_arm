"""Environment rollout with Mahalanobis trace capture and plotting."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import inside_slurm
from tqdm import tqdm

from piper_arm.embedding import embed_prefix_pooled
from piper_arm.mahalanobis import compute_mahalanobis_np

MA_WINDOW = 10


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
            # (C, H, W) float [0,1] -> (H, W, C) uint8
            frame[key] = (val.permute(1, 2, 0) * 255).to(torch.uint8)
        else:
            frame[key] = val

    return frame


def rollout(
    policy: PI05Policy | SmolVLAPolicy,
    vec_env: Any,
    preprocessor: Any,
    postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    gauss_mean: np.ndarray,
    gauss_cov_inv: np.ndarray,
    seeds: list[int],
    desc: str = "",
) -> list[dict[str, Any]]:
    """Run episodes for all envs in parallel, capturing observations, actions,
    and Mahalanobis traces.

    Each env runs independently; collection stops per-env when it terminates.

    Args:
        seeds: One seed per env. len(seeds) must equal the number of envs in vec_env.

    Returns:
        List of dicts (one per env) with keys: success, trace_distances,
        actions (list of Tensor), observations (list of dicts).
    """
    n_envs = len(seeds)
    max_steps = vec_env.call("_max_episode_steps")[0]
    observation, _ = vec_env.reset(seed=seeds)
    policy.reset()

    active = [True] * n_envs
    results: list[dict[str, Any]] = [
        {"success": False, "trace_distances": [], "actions": [], "observations": []}
        for _ in range(n_envs)
    ]

    for _ in tqdm(range(max_steps), desc=desc, leave=False, disable=inside_slurm()):
        if not any(active):
            break

        observation = preprocess_observation(observation)
        observation = add_envs_task(vec_env, observation)
        observation = env_preprocessor(observation)
        raw_obs = deepcopy(observation)
        observation = preprocessor(observation)
        observation = {
            k: v.to(policy.config.device) if isinstance(v, torch.Tensor) else v
            for k, v in observation.items()
        }

        with torch.inference_mode():
            emb = embed_prefix_pooled(policy, observation)
            emb_np = emb.cpu().numpy()
            dists = compute_mahalanobis_np(emb_np, gauss_mean, gauss_cov_inv)
            action = policy.select_action(observation)

        for i in range(n_envs):
            if not active[i]:
                continue
            obs_i = {
                k: v[i].cpu() if isinstance(v, torch.Tensor) else v
                for k, v in raw_obs.items()
            }
            dist_val = float(dists[i])
            obs_i["maha_distance"] = np.array([dist_val], dtype=np.float32)
            results[i]["observations"].append(obs_i)
            results[i]["trace_distances"].append(dist_val)

        action = postprocessor(action)
        for i in range(n_envs):
            if active[i]:
                results[i]["actions"].append(action[i].cpu())

        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action_np = action_transition[ACTION].to("cpu").numpy()

        observation, _, terminated, truncated, info = vec_env.step(action_np)

        if "final_info" in info:
            is_success = info["final_info"].get("is_success")
            if is_success is not None:
                for i in range(n_envs):
                    if active[i] and (bool(terminated[i]) or bool(truncated[i])):
                        val = (
                            is_success[i]
                            if hasattr(is_success, "__len__")
                            else is_success
                        )
                        if hasattr(val, "item"):
                            val = val.item()
                        if val:
                            results[i]["success"] = True

        for i in range(n_envs):
            if active[i] and (bool(terminated[i]) or bool(truncated[i])):
                active[i] = False

    return results


def plot_traces(results: list[dict], output_dir: Path) -> None:
    """Plot per-episode Mahalanobis distance traces colored by success/failure."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(12, 6))

    for result in results:
        steps = np.arange(len(result["trace_distances"]))
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
