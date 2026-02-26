# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Robotic imitation learning system for the Piper robotic arm. Uses Hugging Face's LeRobot framework for training vision-language-action policies (SmolVLA, PI05) and evaluating them in LIBERO simulation with Mahalanobis distance-based out-of-distribution detection.

**Python 3.10 only** (`>=3.10,<3.11`). Uses **UV** as package manager with a frozen lock file.

## Common Commands

```bash
# Training & evaluation (all use UV + LeRobot CLI under the hood)
make train              # Train policy (lerobot-train --config_path configs/train.yaml)
make finetune           # Fine-tune SmolVLA policy
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
uv run pyright                        # Type check (dev dependency)
```

Pre-commit hooks run `ruff --fix` and `ruff-format` automatically. Ruff rules: E, F, I (errors, pyflakes, isort).

No formal test suite exists — testing is done manually via the Makefile commands.

## Architecture

### Core Modules (`piper_arm/`)

- **eval_dist.py** — Main evaluation pipeline. Two-phase approach: (1) embed dataset with VLM prefix → fit Gaussian via Ledoit-Wolf covariance, (2) rollout policy in LIBERO sim computing Mahalanobis distance for OOD detection with moving-average intervention. Supports paired A/B evaluation with McNemar's test.
- **slurm.py** — SLURM job submission via SSH (fabric). Builds sbatch scripts, rsyncs project to cluster, runs in Singularity containers.
- **push_to_hub.py** — Upload trained checkpoints to HuggingFace Hub.
- **gui/app.py** — Flask web UI for monitoring SLURM jobs, GPU availability, streaming logs (SSE).

### Hardware Plugins

Two separate packages registered as LeRobot plugins:
- **lerobot_robot_piper/** — Piper arm as a LeRobot robot environment (PiperConfig, Piper class). Handles RealSense cameras with platform-specific variants (macOS/Linux).
- **lerobot_teleoperator_piper/** — Piper teleoperator interface for recording demonstrations.

### Configuration

YAML configs in `configs/` drive all workflows:
- `train.yaml` — Dataset, policy type, training hyperparameters, W&B logging
- `eval.yaml` — Evaluation settings (policy args via CLI override)
- `play.yaml` / `record.yaml` — Hardware interaction settings

### Deployment

Singularity container defined in `container.def` (python:3.10-slim base, MuJoCo EGL rendering). Targets L40S/L40 GPUs on HTC cluster.
