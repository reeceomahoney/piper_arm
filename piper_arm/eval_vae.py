"""Evaluate a pretrained policy with per-timestep action-chunk VAE scoring.

Rolls out a policy in LIBERO and at each timestep passes the observation
image and a chunk of future actions through a trained ActionChunkVAE to
compute reconstruction loss, KL divergence, and ELBO.

Usage:
    python piper_arm/eval_vae.py \
        --policy-path reece-omahoney/smolvla-libero-256 \
        --vae-checkpoint outputs/action_vae/checkpoints/epoch=49-step=XXXXX.ckpt \
        --n-episodes 1
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION

from piper_arm.train_vae import ActionChunkVAE


@torch.no_grad()
def select_action_with_chunk(
    policy: SmolVLAPolicy, batch: dict
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Like policy.select_action but also returns the full action chunk.

    Returns:
        action: Single action tensor for env stepping.
        action_chunk: (B, n_action_steps, action_dim) on forward-pass steps,
                      None on dequeue steps.
    """
    policy.eval()
    batch = policy._prepare_batch(batch)
    policy._queues = populate_queues(policy._queues, batch, exclude_keys=[ACTION])

    action_chunk = None
    if len(policy._queues[ACTION]) == 0:
        action_chunk = policy._get_action_chunk(batch)
        # Fill the queue (transposed: n_action_steps, batch_size, action_dim)
        policy._queues[ACTION].extend(
            action_chunk.transpose(0, 1)[: policy.config.n_action_steps]
        )

    action = policy._queues[ACTION].popleft()
    return action, action_chunk


@torch.no_grad()
def compute_vae_metrics(vae: ActionChunkVAE, batch: dict) -> dict:
    """Compute VAE reconstruction metrics for an observation batch + action chunk.

    Args:
        vae: Trained ActionChunkVAE model.
        batch: Dict with observation keys, task description, and action chunk,
               in the same format as LeRobotDataset returns (with batch dim).

    Returns:
        Dict with recon_loss, kl_loss, elbo
    """
    recon, actions, mu, logvar = vae(batch)

    recon_loss = F.mse_loss(recon, actions)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    elbo = -(recon_loss + vae.kl_weight * kl_loss)

    return {
        "recon_loss": recon_loss.item(),
        "kl_loss": kl_loss.item(),
        "elbo": elbo.item(),
    }


def plot_vae_stats(results: list[dict], output_dir: Path):
    """Plot per-timestep VAE metrics for each episode."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    metric_keys = [
        ("recon_loss", "Recon Loss (MSE)"),
        ("kl_loss", "KL Divergence"),
        ("elbo", "ELBO"),
    ]

    for record in results:
        steps = [t["step"] for t in record["timesteps"]]
        color = "#2ecc71" if record["success"] else "#e74c3c"
        label = f"T{record['task_id']} E{record['episode']}" + (
            " \u2713" if record["success"] else " \u2717"
        )
        for ax, (key, _) in zip(axes, metric_keys):
            vals = [t[key] for t in record["timesteps"]]
            ax.plot(steps, vals, color=color, alpha=0.7, label=label)

    for ax, (_, title) in zip(axes, metric_keys):
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color="#2ecc71", label="Success"),
        Line2D([0], [0], color="#e74c3c", label="Failure"),
    ]
    axes[0].legend(handles=handles, fontsize=8)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Per-Timestep Action-Chunk VAE Metrics")
    fig.tight_layout()

    plot_path = output_dir / "eval_vae_stats.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Policy rollout with per-timestep action-chunk VAE likelihood"
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default="reece-omahoney/smolvla-libero-256",
        help="HF repo or local path to pretrained policy",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        required=True,
        help="Path to ActionChunkVAE Lightning checkpoint",
    )
    parser.add_argument("--n-episodes", type=int, default=1, help="Episodes per task")
    args = parser.parse_args()

    os.environ["MUJOCO_GL"] = "egl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # ── Load VAE (reuses the same policy for feature extraction) ──
    vae = ActionChunkVAE.load_from_checkpoint(
        args.vae_checkpoint, policy=policy, map_location=device
    ).eval()

    # ── Create environments via make_env ──
    envs = make_env(env_cfg, n_envs=args.n_episodes)

    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = Path(f"outputs/eval_action_vae/{timestamp}")
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

                # Numpy arrays → tensors with LeRobot keys
                observation = preprocess_observation(observation)
                observation = add_envs_task(vec_env, observation)
                observation = env_preprocessor(observation)

                # Save pre-normalized obs for VAE (which has its own preprocessor)
                vae_obs = {
                    k: v.clone().to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in observation.items()
                }

                observation = preprocessor(observation)

                with torch.inference_mode():
                    action, action_chunk = select_action_with_chunk(policy, observation)

                # On forward-pass steps, score the full action chunk with the VAE
                if action_chunk is not None:
                    vae_batch = vae_obs
                    vae_batch["action"] = action_chunk[:, : vae.chunk_size].to(device)

                    metrics = compute_vae_metrics(vae, vae_batch)
                    step_record = {
                        "step": step,
                        "recon_loss": metrics["recon_loss"],
                        "kl_loss": metrics["kl_loss"],
                        "elbo": metrics["elbo"],
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
                mean_recon = np.mean([m["recon_loss"] for m in timestep_metrics])
                mean_kl = np.mean([m["kl_loss"] for m in timestep_metrics])
                mean_elbo = np.mean([m["elbo"] for m in timestep_metrics])
            else:
                mean_recon = mean_kl = mean_elbo = float("nan")

            episode_record = {
                "task_id": task_id,
                "task_description": task_desc,
                "episode": ep,
                "n_steps": step + 1,
                "success": bool(is_success),
                "mean_recon_loss": float(mean_recon),
                "mean_kl_loss": float(mean_kl),
                "mean_elbo": float(mean_elbo),
                "timesteps": timestep_metrics,
            }
            results.append(episode_record)

            print(
                f"  Episode {ep}: {step + 1} steps, success={is_success}, "
                f"mean_recon={mean_recon:.6f}, mean_kl={mean_kl:.4f}, "
                f"mean_elbo={mean_elbo:.6f}"
            )

        vec_env.close()

    # Save results
    output_path = output_dir / "eval_vae_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print overall summary
    successes = [r["success"] for r in results]
    print(f"\nOverall success rate: {sum(successes)}/{len(successes)}")
    print(f"Mean recon loss: {np.mean([r['mean_recon_loss'] for r in results]):.6f}")
    print(f"Mean KL loss: {np.mean([r['mean_kl_loss'] for r in results]):.4f}")
    print(f"Mean ELBO: {np.mean([r['mean_elbo'] for r in results]):.6f}")

    # Plot per-timestep VAE stats
    plot_vae_stats(results, output_dir)


if __name__ == "__main__":
    main()
