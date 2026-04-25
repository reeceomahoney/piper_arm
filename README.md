# DistAL

**Dist**ance-based **A**dvantage **L**earning for robotic imitation learning. A
RECAP-style RL pipeline for training VLAs with advantage-conditioned fine-tuning
and Mahalanobis distance based rewards, built on top of
[LeRobot](https://github.com/huggingface/lerobot).

## Pipeline

The system implements a multi-stage training pipeline:

1. **Collect rollouts** — Roll out and record a base VLA policy in LIBERO.

1. **Compute Mahalanobis stats** — Fit a Ledoit-Wolf mean / inverse covariance
   over mean-pooled VLM image-token embeddings of the base dataset, used both as
   an OOD-aware reward signal and as a failure predictor.

1. **Train value function** — Train a distributional value model using
   cross-entropy over discretized return bins. Reward is either fixed `-1` per
   step or Mahalanobis-based. The code in `distal/train_value.py` and the
   `RECAPValueNetwork` in `distal/value_model.py` was adapted from the upstream
   LeRobot RECAP value-network PR (`jv/recap-value-network` branch).

1. **Fine-tune with advantage conditioning** — Run `distal/train_pi_star.py`,
   which precomputes n-step TD advantages with the frozen value network (cached,
   content-addressed), binarizes them per-task at a percentile threshold, and
   fine-tunes a Pi0.5-based PiStar06 policy with binary advantage conditioning +
   classifier-free guidance dropout.

## Project Structure

```
distal/                             # Core pipeline
├── collect.py                      # Policy rollout → dataset (LIBERO)
├── collect_libero_plus.py          # Same, for LIBERO-plus suite
├── compute_maha_stats.py           # Mahalanobis mean / inv-cov fitting
├── maha_reward.py                  # Mahalanobis → per-step reward
├── maha_auroc.py                   # Mahalanobis as failure predictor (AUROC)
├── train_value.py                  # Distributional value function training
├── train_pi_star.py                # PiStar06 fine-tune (in-script advantage precompute)
├── advantage_cache.py              # Content-addressed advantage cache (HF Hub mirrored)
├── value_model.py                  # RECAPValueNetwork (SmolVLM + expert + categorical head)
├── embedding.py                    # VLM prefix extraction for OOD detection
├── eval_guidance.py                # Sweep CFG guidance scales via lerobot-eval
├── rollout_value_viz.py            # Rerun visualization of rollouts + value estimates
├── push_to_hub.py                  # Upload checkpoints / value networks to HF Hub
├── plotting/                       # Diagnostics: maha rewards, GT returns
└── hardware/                       # Piper arm bring-up (CAN, zeroing)

lerobot_policy_pistar06/            # LeRobot plugin: advantage-conditioned Pi0.5
├── configuration_pistar06.py       # PiStar06Config (extends PI05Config)
└── modeling_pistar06.py            # PiStar06Policy (flat composition, embed_suffix conditioning)

lerobot_robot_piper/                # LeRobot plugin: Piper arm hardware interface
lerobot_teleoperator_piper/         # LeRobot plugin: Piper teleoperator for demos

configs/                            # YAML configs for all workflows
├── train.yaml                      # Base policy training
├── eval.yaml                       # LIBERO evaluation (policy args via CLI)
├── slurm.yaml                      # HPC cluster submission
├── sky.yaml / sky-ssh.yaml         # SkyPilot launch (Vast / SSH)
├── record.yaml                     # Teleop recording
└── play.yaml                       # Hardware playback
```

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Usage

Workflows are driven by [mise](https://mise.jdx.dev/) tasks and YAML configs.
Cluster job submission is managed with
[slurm-tools](https://github.com/reeceomahoney/slurm-tools).

```bash
# Training & evaluation
mise run train              # Train base policy
mise run eval               # Evaluate in LIBERO simulation

# RL pipeline
uv run python -m distal.collect
uv run python -m distal.compute_maha_stats
uv run python -m distal.train_value
uv run python -m distal.train_pi_star

# Hardware
mise run record             # Record demonstrations via teleop
mise run play               # Play trained policy on physical arm

# Cluster / cloud
mise run sky [cluster_id]   # Launch (or sky exec on) SkyPilot cluster
mise run container          # Build Singularity container and upload to HPC
uv run slurm run            # Submit SLURM job
```

## Hardware

- **Piper robotic arm** — 6-DOF + gripper, controlled via CAN bus
- **2x Intel RealSense D435** — wrist-mounted and scene cameras (640x480 @
  30fps)
