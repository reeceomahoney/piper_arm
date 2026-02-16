"""Evaluate flow matching velocity norm as a signal for model confidence.

Rolls out a policy in LIBERO and at each action-chunk forward pass records
the L2 norm of the predicted velocity at each ODE denoising step. The mean
velocity norm across ODE steps is stored per environment timestep.

Usage:
    python piper_arm/eval_velocity_variance.py \
        --policy-path reece-omahoney/smolvla-libero-256 \
        --n-episodes 1
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
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


@torch.no_grad()
def sample_actions_with_velocity_norms(
    policy: SmolVLAPolicy, batch: dict, last_n_steps: int | None = None
) -> dict:
    """Run the ODE solver once, recording ||v_t|| at each denoising step.

    Returns:
        Dict with per-ODE-step velocity norms, their mean, and the action chunk.
    """
    model = policy.model
    device = next(model.parameters()).device

    # Prepare inputs exactly like _get_action_chunk
    for k in batch:
        if k in policy._queues and k != ACTION:
            batch[k] = torch.stack(list(policy._queues[k]), dim=1)

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    # Embed prefix and cache KV
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    _, past_key_values = model.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=model.config.use_cache,
        fill_kv_cache=True,
    )

    bsize = state.shape[0]
    num_steps = model.config.num_steps
    dt = -1.0 / num_steps
    actions_shape = (bsize, model.config.chunk_size, model.config.max_action_dim)

    noise = model.sample_noise(actions_shape, device)
    x_t = noise
    v_norms = []
    velocities = []  # store full vectors for curvature computation

    for step in range(num_steps):
        time = 1.0 + step * dt
        time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(
            bsize
        )

        v_t = model.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=time_tensor,
        )

        # L2 norm of the full velocity vector, averaged over batch
        v_norm = v_t.norm(dim=-1).mean().item()
        v_norms.append(v_norm)
        # Flatten to (B, chunk*dim) for curvature calc
        velocities.append(v_t.reshape(bsize, -1))

        x_t = x_t + dt * v_t

    # Unpad to original action dim
    original_action_dim = policy.config.action_feature.shape[0]
    action_chunk = x_t[:, :, :original_action_dim]

    norms_for_mean = v_norms[-last_n_steps:] if last_n_steps else v_norms

    # Variance of consecutive velocity differences (jerkiness of the ODE trajectory)
    v_diffs = [v_norms[i + 1] - v_norms[i] for i in range(len(v_norms) - 1)]
    v_diff_var = float(np.var(v_diffs)) if len(v_diffs) > 1 else 0.0

    # Curvature of the generation trajectory: κ = ||a_perp|| / ||v||^2
    # where a = (v_{i+1} - v_i) / dt and a_perp is the component of a
    # perpendicular to v.
    curvatures = []
    for i in range(len(velocities) - 1):
        v = velocities[i]  # (B, D)
        a = (velocities[i + 1] - v) / abs(dt)  # (B, D)
        v_norm_sq = (v * v).sum(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, 1)
        v_hat = v / v_norm_sq.sqrt()  # unit velocity
        a_parallel = (a * v_hat).sum(dim=-1, keepdim=True) * v_hat
        a_perp = a - a_parallel
        kappa = a_perp.norm(dim=-1) / v_norm_sq.squeeze(-1)  # (B,)
        curvatures.append(kappa.mean().item())

    mean_curvature = float(np.mean(curvatures)) if curvatures else 0.0

    return {
        "v_norms_per_step": v_norms,
        "mean_v_norm": float(np.mean(norms_for_mean)),
        "v_diff_variance": v_diff_var,
        "mean_curvature": mean_curvature,
        "curvatures_per_step": curvatures,
        "action_chunk": action_chunk,
    }


@torch.no_grad()
def select_action_with_velocity_norms(
    policy: SmolVLAPolicy, batch: dict, last_n_steps: int | None = None
) -> tuple[torch.Tensor, dict | None]:
    """Like select_action but also returns velocity norm stats.

    Returns:
        action: Single action tensor for env stepping.
        stats: Velocity norm dict on forward-pass steps, None on dequeue steps.
    """
    policy.eval()
    batch = policy._prepare_batch(batch)
    policy._queues = populate_queues(policy._queues, batch, exclude_keys=[ACTION])

    stats = None
    if len(policy._queues[ACTION]) == 0:
        stats = sample_actions_with_velocity_norms(policy, batch, last_n_steps)
        action_chunk = stats["action_chunk"]
        policy._queues[ACTION].extend(
            action_chunk.transpose(0, 1)[: policy.config.n_action_steps]
        )

    action = policy._queues[ACTION].popleft()
    return action, stats


def plot_velocity_norms(results: list[dict], output_dir: Path):
    """Plot per-timestep velocity norm, diff variance, and curvature."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for record in results:
        steps = [t["step"] for t in record["timesteps"]]
        color = "#2ecc71" if record["success"] else "#e74c3c"

        norm_vals = [t["mean_v_norm"] for t in record["timesteps"]]
        axes[0].plot(steps, norm_vals, color=color, alpha=0.7)

        diff_vals = [t["v_diff_variance"] for t in record["timesteps"]]
        axes[1].plot(steps, diff_vals, color=color, alpha=0.7)

        curv_vals = [t["mean_curvature"] for t in record["timesteps"]]
        axes[2].plot(steps, curv_vals, color=color, alpha=0.7)

    axes[0].set_ylabel("Mean Velocity Norm")
    axes[1].set_ylabel("Var of Consecutive Velocity Diffs")
    axes[2].set_ylabel("Mean Trajectory Curvature")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color="#2ecc71", label="Success"),
        Line2D([0], [0], color="#e74c3c", label="Failure"),
    ]
    axes[0].legend(handles=handles, fontsize=8)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Flow Matching Velocity Stats per Timestep")
    fig.tight_layout()

    plot_path = output_dir / "eval_velocity_norm.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Policy rollout with flow matching velocity norm tracking"
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default="reece-omahoney/smolvla-libero-256",
        help="HF repo or local path to pretrained policy",
    )
    parser.add_argument("--n-episodes", type=int, default=1, help="Episodes per task")
    parser.add_argument(
        "--last-n-steps",
        type=int,
        default=None,
        help="Only average the last N ODE steps for mean_v_norm (default: all steps)",
    )
    args = parser.parse_args()

    os.environ["MUJOCO_GL"] = "egl"

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

    # ── Create environments ──
    envs = make_env(env_cfg, n_envs=args.n_episodes)

    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = Path(f"outputs/eval_velocity_norm/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for task_id, vec_env in envs[suite_name].items():
        task_desc = vec_env.call("task_description")[0]
        n_tasks = len(envs[suite_name])
        print(f"\n=== Task {task_id}/{n_tasks}: {task_desc} ===")

        max_steps = vec_env.call("_max_episode_steps")[0]

        for ep in range(args.n_episodes):
            observation, info = vec_env.reset(seed=[ep])
            policy.reset()
            is_success = False
            timestep_metrics = []
            done = np.array([False] * vec_env.num_envs)

            for step in range(max_steps):
                if np.all(done):
                    break

                observation = preprocess_observation(observation)
                observation = add_envs_task(vec_env, observation)
                observation = env_preprocessor(observation)
                observation = preprocessor(observation)

                with torch.inference_mode():
                    action, stats = select_action_with_velocity_norms(
                        policy, observation, last_n_steps=args.last_n_steps
                    )

                if stats is not None:
                    step_record = {
                        "step": step,
                        "mean_v_norm": stats["mean_v_norm"],
                        "v_diff_variance": stats["v_diff_variance"],
                        "mean_curvature": stats["mean_curvature"],
                        "v_norms_per_ode_step": stats["v_norms_per_step"],
                        "curvatures_per_ode_step": stats["curvatures_per_step"],
                    }
                    timestep_metrics.append(step_record)

                action = postprocessor(action)

                action_transition = {ACTION: action}
                action_transition = env_postprocessor(action_transition)
                action_np = action_transition[ACTION].to("cpu").numpy()

                observation, reward, terminated, truncated, info = vec_env.step(
                    action_np
                )

                if "final_info" in info:
                    successes = info["final_info"]["is_success"].tolist()
                    is_success = any(successes)

                done = terminated | truncated | done

            # Episode summary
            if timestep_metrics:
                mean_v_norm = np.mean([m["mean_v_norm"] for m in timestep_metrics])
            else:
                mean_v_norm = float("nan")

            episode_record = {
                "task_id": task_id,
                "task_description": task_desc,
                "episode": ep,
                "n_steps": step + 1,
                "success": bool(is_success),
                "mean_velocity_norm": float(mean_v_norm),
                "timesteps": timestep_metrics,
            }
            results.append(episode_record)

            print(
                f"  Episode {ep}: {step + 1} steps, success={is_success}, "
                f"mean_v_norm={mean_v_norm:.6f}"
            )

        vec_env.close()

    # Save results
    output_path = output_dir / "eval_velocity_norm_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print overall summary
    successes = [r["success"] for r in results]
    print(f"\nOverall success rate: {sum(successes)}/{len(successes)}")
    print(
        f"Mean velocity norm: {np.mean([r['mean_velocity_norm'] for r in results]):.6f}"
    )

    # Plot
    plot_velocity_norms(results, output_dir)


if __name__ == "__main__":
    main()
