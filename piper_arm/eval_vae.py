"""Evaluate a pretrained policy with per-timestep VAE likelihood scoring.

Rolls out a policy in LIBERO and at each timestep passes the observation
image through a trained ImageVAE to compute reconstruction loss, KL
divergence, and ELBO — measuring how "in-distribution" each observation is.

Usage:
    python piper_arm/eval_vae.py \
        --policy-path reece-omahoney/smolvla-libero \
        --vae-checkpoint outputs/vae/checkpoints/epoch=49-step=XXXXX.ckpt \
        --suite-name libero_object \
        --n-episodes 1
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv, _get_suite
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from torchvision.transforms.functional import resize
from transformers import AutoModelForImageTextToText

from piper_arm.train_vae import ImageVAE


def add_batch_dim(obs: dict) -> dict:
    """Recursively add a leading batch dimension to all numpy arrays."""
    result = {}
    for k, v in obs.items():
        if isinstance(v, dict):
            result[k] = add_batch_dim(v)
        elif isinstance(v, np.ndarray):
            result[k] = v[np.newaxis]
        else:
            result[k] = v
    return result


def load_vae(checkpoint_path: str, device: torch.device) -> ImageVAE:
    """Load a trained ImageVAE from a Lightning checkpoint.

    The frozen vision_model and connector are not saved in the checkpoint,
    so we load them fresh from the SmolVLM weights and pass them in.
    """
    VLM_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    print(f"Loading VLM for VAE: {VLM_MODEL}")
    vlm = AutoModelForImageTextToText.from_pretrained(VLM_MODEL, dtype=torch.bfloat16)
    vision_model = vlm.model.vision_model
    connector = vlm.model.connector
    connector_dim = vlm.config.text_config.hidden_size
    del vlm

    vae = ImageVAE.load_from_checkpoint(
        checkpoint_path,
        vision_model=vision_model,
        connector=connector,
        connector_dim=connector_dim,
    )
    vae.to(device)
    vae.eval()
    return vae


@torch.no_grad()
def compute_vae_metrics(vae: ImageVAE, image: torch.Tensor, kl_weight: float) -> dict:
    """Compute VAE reconstruction metrics for a single image.

    Args:
        vae: Trained ImageVAE model.
        image: (1, C, H, W) tensor in [0, 1].
        kl_weight: KL weight matching VAE training.

    Returns:
        Dict with recon_loss, kl_loss, elbo, mu_mean, mu_std, logvar_mean,
        and the reconstruction tensor.
    """
    # Normalize to [-1, 1] for the vision encoder
    pixel_values = (image - 0.5) / 0.5
    recon, mu, logvar = vae(pixel_values)

    # Reconstruction target is [0, 1]
    recon_loss = F.mse_loss(recon, image)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    elbo = -(recon_loss + kl_weight * kl_loss)

    return {
        "recon_loss": recon_loss.item(),
        "kl_loss": kl_loss.item(),
        "elbo": elbo.item(),
        "mu_mean": mu.mean().item(),
        "mu_std": mu.std().item(),
        "logvar_mean": logvar.mean().item(),
        "recon": recon,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Policy rollout with per-timestep VAE likelihood"
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
        help="Path to ImageVAE Lightning checkpoint",
    )
    parser.add_argument(
        "--suite-name", type=str, default="libero_10", help="LIBERO task suite"
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="*",
        default=None,
        help="Specific task IDs (default: all)",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        default="observation.images.image",
        help="Camera key for VAE input",
    )
    parser.add_argument("--n-episodes", type=int, default=1, help="Episodes per task")
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=1e-4,
        help="KL weight matching VAE training",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/eval_vae", help="Output directory"
    )
    parser.add_argument(
        "--save-recon",
        action="store_true",
        help="Save reconstruction images every 10 steps",
    )
    args = parser.parse_args()

    os.environ["MUJOCO_GL"] = "egl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load policy ──
    env_cfg = LiberoEnvConfig(args.suite_name)
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

    # ── Load VAE ──
    vae = load_vae(args.vae_checkpoint, device)

    # ── Setup ──
    suite = _get_suite(args.suite_name)
    n_tasks = len(suite.tasks)
    max_steps = TASK_SUITE_MAX_STEPS.get(args.suite_name, 280)
    task_ids = args.task_ids if args.task_ids is not None else list(range(n_tasks))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_recon:
        recon_dir = output_dir / "reconstructions"
        recon_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for task_id in task_ids:
        task = suite.get_task(task_id)
        print(f"\n=== Task {task_id}/{n_tasks}: {task.language} ===")

        env = LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=args.suite_name,
            obs_type="pixels_agent_pos",
            camera_name="agentview_image,robot0_eye_in_hand_image",
            init_states=True,
            episode_index=0,
            control_mode="relative",
        )

        for ep in range(args.n_episodes):
            env._init_state_id = ep
            obs, info = env.reset()
            policy.reset()
            is_success = False
            timestep_metrics = []

            for step in range(max_steps):
                # Process observation for policy
                batched_obs = add_batch_dim(obs)
                policy_obs = preprocess_observation(batched_obs)

                # Extract image for VAE before policy preprocessor runs
                image = policy_obs[args.camera_key].to(device)  # (1, C, H, W) [0,1]
                if image.shape[-2:] != (256, 256):
                    image = resize(image, [256, 256])

                # Compute VAE metrics
                metrics = compute_vae_metrics(vae, image, args.kl_weight)
                step_record = {
                    "step": step,
                    "recon_loss": metrics["recon_loss"],
                    "kl_loss": metrics["kl_loss"],
                    "elbo": metrics["elbo"],
                    "mu_mean": metrics["mu_mean"],
                    "mu_std": metrics["mu_std"],
                    "logvar_mean": metrics["logvar_mean"],
                }
                timestep_metrics.append(step_record)

                # Save reconstruction images periodically
                if args.save_recon and step % 10 == 0:
                    from torchvision.utils import save_image

                    pair = torch.cat([image, metrics["recon"]], dim=0)
                    save_image(
                        pair,
                        recon_dir / f"task{task_id}_ep{ep}_step{step}.png",
                        nrow=2,
                    )

                # Run policy
                policy_obs["task"] = [env.task_description]
                policy_obs = env_preprocessor(policy_obs)
                policy_obs = preprocessor(policy_obs)

                with torch.inference_mode():
                    action = policy.select_action(policy_obs)
                action = postprocessor(action)

                action_transition = {ACTION: action}
                action_transition = env_postprocessor(action_transition)
                action_np = action_transition[ACTION].squeeze(0).cpu().numpy()

                # Step env
                raw_obs, reward, done, step_info = env._env.step(action_np)
                is_success = env._env.check_success()
                terminated = done or is_success

                if terminated:
                    break

                obs = env._format_raw_obs(raw_obs)

            # Episode summary
            mean_recon = np.mean([m["recon_loss"] for m in timestep_metrics])
            mean_kl = np.mean([m["kl_loss"] for m in timestep_metrics])
            mean_elbo = np.mean([m["elbo"] for m in timestep_metrics])

            episode_record = {
                "task_id": task_id,
                "task_description": task.language,
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

        env.close()

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

    # Deduplicated legend showing only success/failure colors
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color="#2ecc71", label="Success"),
        Line2D([0], [0], color="#e74c3c", label="Failure"),
    ]
    axes[0].legend(handles=handles, fontsize=8)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Per-Timestep VAE Metrics")
    fig.tight_layout()

    plot_path = output_dir / "eval_vae_stats.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
