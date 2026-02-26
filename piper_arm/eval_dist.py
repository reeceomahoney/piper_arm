"""Evaluate Mahalanobis distance of VLM prefix embeddings during sim rollouts.

Supports both PI05 and SmolVLA policies (auto-detected from checkpoint).

Phase 1: Embed the entire training dataset to fit a Gaussian (mean + covariance)
          over the VLM prefix representations. These stats can be cached to disk.

Phase 2: Roll out the policy in LIBERO and at each timestep compute the VLM
          prefix embedding and its Mahalanobis distance from the training
          distribution. Plot the distance over each trajectory.

          With --intervene: when the moving average of distances exceeds
          INTERVENTION_K times the baseline (mean of the first MA_WINDOW
          forward-pass distances), the new action chunk is blended with the
          previous chunk (weighted by BLEND_ALPHA) to dampen erratic
          predictions. Action chunk length is always n_action_steps.

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
import os
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
INTERVENTION_THRESHOLD = 37.0

CONDITIONS = ["control", "intervene"]


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
    prev_chunk: list[Optional[torch.Tensor]],
    intervene: bool = False,
) -> tuple[torch.Tensor, dict | None]:
    """Like select_action but also returns Mahalanobis distance of the current obs.

    Supports both PI05 and SmolVLA policies.

    When intervene=True and OOD is detected, the previous action chunk is
    reversed (negated deltas played in reverse order) to backtrack the robot
    to its pre-chunk state, then the policy re-plans on the next forward pass.
    Action chunk length is always n_action_steps.

    Args:
        dist_histories: Per-episode list of distances from forward passes so far
                        this episode. Updated in-place. Reset between episodes.
        prev_chunk: Single-element list holding the previous action chunk tensor
                    (mutated in-place). Pass [None] at episode start.

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

        # Intervene if any episode's moving average exceeds absolute threshold
        ood = False
        if intervene:
            for history in dist_histories:
                if len(history) >= MA_WINDOW:
                    current_ma = np.mean(history[-MA_WINDOW:])
                    if current_ma > INTERVENTION_THRESHOLD:
                        ood = True
                        break

        stats = {"mahalanobis": dist.tolist(), "intervention": ood}

        # Backtrack: reverse the previous chunk's deltas to undo recent motion,
        # then let the next forward pass re-plan from the recovered state.
        if ood and prev_chunk[0] is not None:
            reversed_chunk = -prev_chunk[0].flip(dims=[1])
            action_queue.extend(reversed_chunk.transpose(0, 1))
            prev_chunk[0] = None
        else:
            action_chunk = policy.predict_action_chunk(batch)
            n_steps = policy.config.n_action_steps
            action_chunk = action_chunk[:, :n_steps]
            prev_chunk[0] = action_chunk
            action_queue.extend(action_chunk.transpose(0, 1))

    action = action_queue.popleft()
    return action, stats


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────


def _plot_condition(
    ax: Any,
    paired_results: list[dict],
    condition: str,
    title: str,
):
    """Plot Mahalanobis traces for one condition (control or intervene)."""
    from matplotlib.lines import Line2D

    success_key = f"{condition}_success"
    timesteps_key = f"{condition}_timesteps"

    has_interventions = condition == "intervene" and any(
        any(t.get("intervention") for t in record[timesteps_key])
        for record in paired_results
    )

    for record in paired_results:
        timesteps = record[timesteps_key]
        if not timesteps:
            continue
        steps = np.array([t["step"] for t in timesteps])
        dists = np.array([t["mahalanobis"] for t in timesteps])
        iv_flags = np.array([bool(t.get("intervention")) for t in timesteps])
        color = "#2ecc71" if record[success_key] else "#e74c3c"

        if len(dists) >= MA_WINDOW:
            kernel = np.ones(MA_WINDOW) / MA_WINDOW
            ma_dists = np.convolve(dists, kernel, mode="valid")
            ma_steps = steps[MA_WINDOW - 1 :]
            ma_iv = np.convolve(iv_flags.astype(float), kernel, mode="valid") > 0
        else:
            ma_dists = dists
            ma_steps = steps
            ma_iv = iv_flags

        if not ma_iv.any():
            ax.plot(ma_steps, ma_dists, color=color, alpha=0.7, linewidth=1.0)
        else:
            # Draw segments, switching style when intervention state changes
            changes = np.where(np.diff(ma_iv.astype(int)) != 0)[0] + 1
            segments = np.split(np.arange(len(ma_steps)), changes)
            for seg in segments:
                if len(seg) == 0:
                    continue
                # Extend by 1 point to connect segments seamlessly
                lo = seg[0]
                hi = min(seg[-1] + 2, len(ma_steps))
                style = ":" if ma_iv[seg[0]] else "-"
                ax.plot(
                    ma_steps[lo:hi],
                    ma_dists[lo:hi],
                    color=color,
                    alpha=0.7,
                    linewidth=1.0,
                    linestyle=style,
                )

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
    ax.set_ylabel(f"Mahalanobis Distance (MA, w={MA_WINDOW})")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_mahalanobis(paired_results: list[dict], output_dir: Path):
    """Plot paired A/B Mahalanobis distance traces side by side."""
    import matplotlib.pyplot as plt

    fig, (ax_ctrl, ax_intv) = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    _plot_condition(ax_ctrl, paired_results, "control", "Control (no intervention)")
    _plot_condition(ax_intv, paired_results, "intervene", "With Intervention")

    fig.suptitle(
        "Paired A/B: Mahalanobis Distance of VLM Prefix Embeddings",
        fontsize=14,
    )
    fig.tight_layout()

    plot_path = output_dir / "eval_dist.png"
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


@dataclass
class EvalMahalanobisConfig:
    policy_path: str = "reece-omahoney/smolvla-libero-16-chunk"
    dataset: str = "reece-omahoney/libero"
    n_episodes: int = 50
    batch_size: int = 32
    num_workers: int = 8
    load_stats: Optional[str] = os.path.join(
        os.environ.get("OUTPUT_DIR", "outputs"),
        "eval_dist/2026-02-24/17-07-42/gauss_stats.npz",
    )


def _run_episode(
    policy: Union[PI05Policy, SmolVLAPolicy],
    vec_env: Any,
    preprocessor: Any,
    postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    gauss_mean: np.ndarray,
    gauss_cov_inv: np.ndarray,
    seed: int,
    intervene: bool,
    desc: str = "",
) -> dict[str, Any]:
    """Run a single episode and return the result record."""
    max_steps = vec_env.call("_max_episode_steps")[0]  # type: ignore[attr-defined]
    observation, info = vec_env.reset(seed=[seed])  # type: ignore[arg-type]
    policy.reset()
    success = False
    timestep_metrics: list[dict[str, Any]] = []
    dist_history: list[float] = []
    prev_chunk: list[Optional[torch.Tensor]] = [None]
    done = False
    step = 0

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
        observation = preprocessor(observation)

        with torch.inference_mode():
            action, stats = select_action_with_mahalanobis(
                policy,
                observation,
                gauss_mean,
                gauss_cov_inv,
                dist_histories=[dist_history],
                done=np.array([done]),
                prev_chunk=prev_chunk,
                intervene=intervene,
            )

        if stats is not None:
            entry: dict[str, Any] = {
                "step": step,
                "mahalanobis": stats["mahalanobis"][0],
            }
            if intervene:
                entry["intervention"] = stats["intervention"]
            timestep_metrics.append(entry)

        action = postprocessor(action)

        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action_np = action_transition[ACTION].to("cpu").numpy()

        observation, _, terminated, truncated, info = vec_env.step(action_np)

        if "final_info" in info:
            if info["final_info"]["is_success"].item():
                success = True

        done = bool(terminated | truncated)

    if timestep_metrics:
        mean_dist = np.mean([m["mahalanobis"] for m in timestep_metrics])
        n_interventions = sum(
            1 for m in timestep_metrics if m.get("intervention", False)
        )
    else:
        mean_dist = float("nan")
        n_interventions = 0

    return {
        "n_steps": step + 1,
        "success": success,
        "mean_mahalanobis": float(mean_dist),
        "n_interventions": n_interventions,
        "timesteps": timestep_metrics,
    }


def _mcnemar_test(
    pairs: list[dict],
) -> dict[str, Any]:
    """Compute McNemar's test from paired (control, intervene) success outcomes.

    Returns dict with contingency table counts, chi-squared statistic, and p-value.
    """
    from scipy.stats import binomtest

    # Contingency table for discordant pairs
    # b = control fail, treatment success (intervention helped)
    # c = control success, treatment fail (intervention hurt)
    b = sum(1 for p in pairs if not p["control_success"] and p["intervene_success"])
    c = sum(1 for p in pairs if p["control_success"] and not p["intervene_success"])
    a = sum(1 for p in pairs if p["control_success"] and p["intervene_success"])
    d = sum(1 for p in pairs if not p["control_success"] and not p["intervene_success"])

    n_discordant = b + c
    if n_discordant == 0:
        p_value = 1.0
        chi2 = 0.0
    elif n_discordant < 25:
        # Exact binomial test for small samples
        p_value = float(binomtest(b, n_discordant, 0.5).pvalue)
        chi2 = float("nan")
    else:
        # McNemar's chi-squared with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        from scipy.stats import chi2 as chi2_dist

        p_value = float(1 - chi2_dist.cdf(chi2, df=1))

    return {
        "concordant_both_success": a,
        "concordant_both_fail": d,
        "discordant_intervention_helped": b,
        "discordant_intervention_hurt": c,
        "n_discordant": n_discordant,
        "chi2": chi2,
        "p_value": p_value,
    }


@draccus.wrap()  # type: ignore[misc]
def main(cfg: EvalMahalanobisConfig):
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
    output_base = Path(os.environ.get("OUTPUT_DIR", "outputs"))
    output_dir = output_base / f"eval_dist/{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Gaussian stats for reuse
    np.savez(output_dir / "gauss_stats.npz", mean=gauss_mean, cov_inv=gauss_cov_inv)
    print(f"Saved Gaussian stats to {output_dir / 'gauss_stats.npz'}")

    # ── Phase 2: Paired A/B Rollout ──
    paired_results: list[dict[str, Any]] = []

    for task_id, vec_env in envs[suite_name].items():
        task_desc = vec_env.call("task_description")[0]  # type: ignore[attr-defined]
        n_tasks = len(envs[suite_name])
        print(f"\n=== Task {task_id + 1}/{n_tasks}: {task_desc} ===")

        for ep in range(cfg.n_episodes):
            pair: dict[str, Any] = {
                "task_id": task_id,
                "task_description": task_desc,
                "episode": ep,
            }

            for condition in CONDITIONS:
                intervene = condition == "intervene"
                label = "intervene" if intervene else "control"

                result = _run_episode(
                    policy=policy,
                    vec_env=vec_env,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    gauss_mean=gauss_mean,
                    gauss_cov_inv=gauss_cov_inv,
                    seed=ep,
                    intervene=intervene,
                    desc=f"  Ep {ep} ({label})",
                )

                pair[f"{condition}_success"] = result["success"]
                pair[f"{condition}_n_steps"] = result["n_steps"]
                pair[f"{condition}_mean_mahalanobis"] = result["mean_mahalanobis"]
                pair[f"{condition}_n_interventions"] = result.get("n_interventions", 0)
                pair[f"{condition}_timesteps"] = result["timesteps"]

            # Print paired result
            ctrl = "OK" if pair["control_success"] else "FAIL"
            intv = "OK" if pair["intervene_success"] else "FAIL"
            print(f"  Episode {ep}: control={ctrl}, intervene={intv}")

            paired_results.append(pair)

        vec_env.close()

    # ── Save results ──
    output_path = output_dir / "eval_dist_results.json"
    with open(output_path, "w") as f:
        json.dump(paired_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # ── Paired Statistical Analysis ──
    n_total = len(paired_results)
    ctrl_successes = sum(1 for p in paired_results if p["control_success"])
    intv_successes = sum(1 for p in paired_results if p["intervene_success"])

    print(f"\n{'=' * 60}")
    print("PAIRED A/B RESULTS")
    print(f"{'=' * 60}")
    print(f"Total paired episodes: {n_total}")
    print(
        f"Control success rate:    {ctrl_successes}/{n_total} "
        f"({100 * ctrl_successes / n_total:.1f}%)"
    )
    print(
        f"Intervene success rate:  {intv_successes}/{n_total} "
        f"({100 * intv_successes / n_total:.1f}%)"
    )
    print(f"Delta: {100 * (intv_successes - ctrl_successes) / n_total:+.1f}pp")

    # McNemar's test
    mcnemar = _mcnemar_test(paired_results)
    print("\nMcNemar's Test (paired binary outcomes):")
    print(f"  Both succeed:             {mcnemar['concordant_both_success']}")
    print(f"  Both fail:                {mcnemar['concordant_both_fail']}")
    print(f"  Intervention helped:      {mcnemar['discordant_intervention_helped']}")
    print(f"  Intervention hurt:        {mcnemar['discordant_intervention_hurt']}")
    print(f"  Total discordant pairs:   {mcnemar['n_discordant']}")
    if not np.isnan(mcnemar["chi2"]):
        print(f"  Chi-squared:              {mcnemar['chi2']:.4f}")
    print(f"  p-value:                  {mcnemar['p_value']:.4f}")

    if mcnemar["p_value"] < 0.05:
        if (
            mcnemar["discordant_intervention_helped"]
            > mcnemar["discordant_intervention_hurt"]
        ):
            print("  → Statistically significant: intervention HELPS (p < 0.05)")
        else:
            print("  → Statistically significant: intervention HURTS (p < 0.05)")
    else:
        print(f"  → Not statistically significant (p = {mcnemar['p_value']:.3f})")

    # Per-task breakdown
    task_ids = sorted(set(p["task_id"] for p in paired_results))
    print("\nPer-task breakdown:")
    print(f"  {'Task':<50} {'Ctrl':>5} {'Intv':>5} {'Delta':>6}")
    print(f"  {'-' * 50} {'-' * 5} {'-' * 5} {'-' * 6}")
    for tid in task_ids:
        task_pairs = [p for p in paired_results if p["task_id"] == tid]
        desc = task_pairs[0]["task_description"]
        n = len(task_pairs)
        c_s = sum(1 for p in task_pairs if p["control_success"])
        i_s = sum(1 for p in task_pairs if p["intervene_success"])
        delta = i_s - c_s
        print(f"  {desc[:50]:<50} {c_s:>2}/{n:<2} {i_s:>2}/{n:<2} {delta:>+4}")

    # Intervention detection (within intervention arm)
    intv_fail_detected = sum(
        1
        for p in paired_results
        if not p["intervene_success"] and p["intervene_n_interventions"] > 0
    )
    intv_fail_missed = sum(
        1
        for p in paired_results
        if not p["intervene_success"] and p["intervene_n_interventions"] == 0
    )
    intv_success_triggered = sum(
        1
        for p in paired_results
        if p["intervene_success"] and p["intervene_n_interventions"] > 0
    )
    intv_success_clean = sum(
        1
        for p in paired_results
        if p["intervene_success"] and p["intervene_n_interventions"] == 0
    )
    n_intv_fail = intv_fail_detected + intv_fail_missed
    n_intv_success = intv_success_triggered + intv_success_clean
    print("\nIntervention detection (within intervention arm):")
    print(f"  Failed episodes detected:          {intv_fail_detected}/{n_intv_fail}")
    print(f"  Failed episodes missed:            {intv_fail_missed}/{n_intv_fail}")
    print(
        f"  Successful episodes triggered:     "
        f"{intv_success_triggered}/{n_intv_success}"
    )
    print(f"  Successful episodes clean:         {intv_success_clean}/{n_intv_success}")

    # Save statistical summary
    summary = {
        "n_total": n_total,
        "control_success_rate": ctrl_successes / n_total,
        "intervene_success_rate": intv_successes / n_total,
        "delta_pp": (intv_successes - ctrl_successes) / n_total * 100,
        "mcnemar": mcnemar,
    }
    with open(output_dir / "statistical_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Plot ──
    plot_mahalanobis(paired_results, output_dir)


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
