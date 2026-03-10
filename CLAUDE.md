# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Robotic imitation learning system for the Piper robotic arm. Uses Hugging Face's LeRobot framework for training vision-language-action policies (SmolVLA, PI05) and evaluating them in LIBERO simulation. Includes a RECAP-style RL pipeline: value function training, advantage estimation, and advantage-conditioned policy fine-tuning, plus Mahalanobis distance-based OOD detection.

**Python 3.10 only** (`>=3.10,<3.11`). Uses **UV** as package manager with a frozen lock file.

## Common Commands

```bash
# Training & evaluation (all use UV + LeRobot CLI under the hood)
make train              # Train policy (lerobot-train --config_path configs/train.yaml)
make finetune           # Fine-tune SmolVLA policy
make train-value        # Train distributional value function
make train-advantage    # Advantage-conditioned policy fine-tuning
make eval               # Evaluate policy in LIBERO sim

# Hardware
make record             # Record demonstrations via teleop
make play               # Play trained policy on physical arm

# Cluster
make container          # Build Singularity container and upload to cluster
uv run slurm            # Submit SLURM job to HTC cluster (see slurm.py)

# GUI (Flask-based SLURM monitor)
make gui-start          # Start background Flask server
make gui-stop           # Stop Flask server

# Code quality
uv run ruff check piper_arm/          # Lint
uv run ruff format piper_arm/         # Format
uv run ty                             # Type check (dev dependency)
```

Pre-commit hooks run `ruff --fix` and `ruff-format` automatically. Ruff rules: E, F, I (errors, pyflakes, isort).

**Never start function or variable names with underscores.** Use plain names without leading underscores for all functions and variables.

No formal test suite exists — testing is done manually via the Makefile commands.

## Architecture

### Training Pipeline

The system implements a RECAP-style RL pipeline:

1. **eval_dist.py** — Evaluate policy in LIBERO sim, create LeRobot dataset with `maha_distance`, `steps_remaining`, `success` fields.
2. **train_value.py** — Train distributional value function (SmolVLM + expert backbone, cross-entropy over discretized return bins).
3. **compute_advantage_labels.py** — Pre-compute binary advantage labels using the trained value model. Computes per-task advantage thresholds (30th percentile), binarizes per-sample advantages, and writes an `advantage_label` column directly into the dataset's parquet files.
4. **lerobot-train (advantage_train.yaml)** — Fine-tune SmolVLA with binary advantage conditioning using the pre-labelled dataset. 30% advantage token dropout for classifier-free guidance.

### Supporting Modules (`piper_arm/`)

- **value_model.py** — `ValueModel` class: SmolVLM + expert with learnable value query token and categorical value head. Vision encoder frozen, VLM + expert trainable.
- **embedding.py** — VLM prefix extraction for PI05/SmolVLA. Mean-pooled embeddings over image tokens.
- **mahalanobis.py** — Mahalanobis distance computation and Gaussian fitting (Ledoit-Wolf covariance).
- **rollout.py** — Single episode rollout with Mahalanobis trace capture and moving-average smoothing.
- **visualize.py** — Rerun-based visualization of rollout traces synced with MP4 videos.
- **slurm.py** — SLURM job submission via SSH (fabric). Builds sbatch scripts, rsyncs project to cluster, runs in Singularity containers.
- **push_to_hub.py** — Upload trained checkpoints to HuggingFace Hub.
- **gui/app.py** — Flask web UI for monitoring SLURM jobs, GPU availability, streaming logs (SSE).
- **zero.py** — Piper arm initialization/zeroing utility.

### Hardware Plugins

Two separate packages registered as LeRobot plugins:

- **lerobot_robot_piper/** — Piper arm as a LeRobot robot environment (PiperConfig, Piper class). Handles RealSense cameras with platform-specific variants (macOS/Linux).
- **lerobot_teleoperator_piper/** — Piper teleoperator interface for recording demonstrations.

### Configuration

YAML configs in `configs/` drive all workflows:

- `train.yaml` — Dataset, policy type, training hyperparameters, W&B logging
- `eval.yaml` — Evaluation settings (policy args via CLI override)
- `advantage_train.yaml` — Advantage-conditioned policy fine-tuning (policy repo, advantage dropout, eval settings)
- `play.yaml` / `record.yaml` — Hardware interaction settings

### Deployment

Singularity container defined in `container.def` (python:3.10-slim base, MuJoCo EGL rendering). Targets L40S/H100 GPUs on HTC cluster.
