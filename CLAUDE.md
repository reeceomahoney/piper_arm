# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Robotic imitation learning system for the Piper robotic arm. Uses Hugging Face's LeRobot framework for training vision-language-action policies (SmolVLA, PI05) and evaluating them in LIBERO simulation. Includes a RECAP-style RL pipeline: value function training, advantage estimation, and advantage-conditioned policy fine-tuning, plus Mahalanobis distance-based OOD detection.

**Python 3.12** (`>=3.12,<3.13`). Uses **UV** as package manager with a frozen lock file. **Mise** as task runner.

## Common Commands

```bash
# Training & evaluation (all use UV + LeRobot CLI under the hood)
mise run train              # Train policy (lerobot-train --config_path configs/train.yaml)
mise run finetune           # Fine-tune SmolVLA policy
mise run train-advantage    # Advantage-conditioned policy fine-tuning
mise run eval               # Evaluate policy in LIBERO sim

# Hardware
mise run record             # Record demonstrations via teleop
mise run play               # Play trained policy on physical arm

# Cluster
mise run container          # Build Singularity container and upload to cluster
uv run slurm run            # Submit SLURM job to HTC cluster (see slurm-tools package)

# GUI (Flask-based SLURM monitor, via slurm-tools package)
uv run slurm gui            # Start background Flask server
uv run slurm gui stop       # Stop Flask server

# Code quality
uv run pre-commit run --all-files     # Run all pre-commit hooks (ruff --fix, ruff-format)
```

Ruff rules: E, F, I (errors, pyflakes, isort).

**Never start function or variable names with underscores.** Use plain names without leading underscores for all functions and variables.

**Do not add `Usage:` sections to module-level docstrings.** Scripts use draccus configs which are self-documenting.

No formal test suite exists — testing is done manually via mise tasks.

slurm_tools has its own git repo, use this to push changes

## Architecture

### Training Pipeline

The system implements a RECAP-style RL pipeline:

1. **collect.py** — Roll out policy in LIBERO sim using LeRobot's `eval_policy()`, create LeRobot dataset with `maha_distance`, `steps_remaining`, `success` fields.
2. **train_value.py** — Train distributional value function (SmolVLM + expert backbone, cross-entropy over discretized return bins).
3. **compute_advantage_labels.py** — Pre-compute binary advantage labels using the trained value model. Computes per-task advantage thresholds (30th percentile), binarizes per-sample advantages, and writes an `advantage_label` column directly into the dataset's parquet files.
4. **lerobot-train (advantage_train.yaml)** — Fine-tune SmolVLA with binary advantage conditioning via the `lerobot_policy_advantage` plugin. 30% advantage token dropout for classifier-free guidance.

### Advantage Policy Plugin (`lerobot_policy_advantage/`)

LeRobot plugin that registers the "advantage" policy type:

- **modeling_advantage.py** — `AdvantagePolicy`: wraps SmolVLA with learned advantage embeddings and handles advantage label loading from dataset.
- **configuration_advantage.py** — `AdvantageConfig` dataclass for policy configuration.

### Supporting Modules (`distal/`)

- **value_model.py** — `ValueModel` class: SmolVLM + expert with learnable value query token and categorical value head. Vision encoder frozen, VLM + expert trainable.
- **embedding.py** — VLM prefix extraction for PI05/SmolVLA. Mean-pooled embeddings over image tokens.
- **mahalanobis.py** — Mahalanobis distance computation and Gaussian fitting (Ledoit-Wolf covariance).
- **add_labels.py** — Add `steps_remaining` and `success` columns to LeRobot datasets.
- **rollout_value_viz.py** — Roll out base policy in LIBERO with value estimates and Rerun visualization.
- **visualize.py** — Rerun-based visualization of rollout traces synced with MP4 videos.
- **push_to_hub.py** — Upload trained checkpoints to HuggingFace Hub.
- **zero.py** — Piper arm initialization/zeroing utility.

### SLURM Tools (`slurm_tools/`)

Separate local package for cluster job management:

- **slurm.py** — SLURM job submission via SSH (fabric). Builds sbatch scripts, rsyncs project to cluster, runs in Singularity containers. Also manages the GUI daemon (start/stop).
- **gui/app.py** — Flask web UI for monitoring SLURM jobs, GPU availability, streaming logs (SSE).

### Hardware Plugins

Two separate packages registered as LeRobot plugins:

- **lerobot_robot_piper/** — Piper arm as a LeRobot robot environment (PiperConfig, Piper class). Handles RealSense cameras with platform-specific variants (macOS/Linux).
- **lerobot_teleoperator_piper/** — Piper teleoperator interface for recording demonstrations.

### Configuration

YAML configs in `configs/` drive all workflows:

- `train.yaml` — Dataset, policy type, training hyperparameters, W&B logging
- `eval.yaml` — Evaluation settings (policy args via CLI override)
- `advantage_train.yaml` — Advantage-conditioned policy fine-tuning (policy repo, advantage dropout, eval settings)
- `slurm.yaml` — SLURM job submission settings (cluster paths, resources, container config)
- `play.yaml` / `record.yaml` — Hardware interaction settings

### Deployment

Singularity container defined in `container.def` (MuJoCo EGL rendering). Targets L40S/H100 GPUs on HTC cluster. Uses a custom LeRobot fork (`reeceomahoney/lerobot@fix/rollout-return-observations-nested-dict`).
