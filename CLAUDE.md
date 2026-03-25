# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

Robotic imitation learning system for the Piper robotic arm. Uses Hugging Face's
LeRobot framework for training vision-language-action policies (SmolVLA, PI05)
and evaluating them in LIBERO simulation. Includes a RECAP-style RL pipeline:
value function training, advantage estimation, and advantage-conditioned policy
fine-tuning, plus Mahalanobis distance-based OOD detection.

**Python 3.12** (`>=3.12,<3.13`). Uses **UV** as package manager with a frozen
lock file. **Mise** as task runner.

## Common Commands

```bash
# Training & evaluation (all use UV + LeRobot CLI under the hood)
mise run train              # Train base policy
mise run adv-train          # Train advantage-conditioned policy
mise run eval               # Evaluate policy in LIBERO sim

# Hardware
mise run record             # Record demonstrations via teleop
mise run play               # Play trained policy on physical arm

# Cloud / Cluster
mise run sky                # Launch training on cloud via SkyPilot
mise run sky-exec           # Send command to running SkyPilot cluster
mise run container          # Build Singularity container and upload to cluster
uv run slurm run            # Submit SLURM job to HTC cluster (see slurm-tools package)

# GUI (Flask-based SLURM monitor, via slurm-tools package)
uv run slurm gui            # Start background Flask server
uv run slurm gui stop       # Stop Flask server

# Code quality
uv run pre-commit run --all-files     # Run all pre-commit hooks (ruff, mdformat, ty)
```

Ruff rules: E, F, I (errors, pyflakes, isort). Pre-commit also runs mdformat
(80-char wrap) and ty (type checking).

**Never start function or variable names with underscores.** Use plain names
without leading underscores for all functions and variables.

**Do not add `Usage:` sections to module-level docstrings.** Scripts use draccus
configs which are self-documenting.

No formal test suite exists — testing is done manually via mise tasks.

**Never use OSMesa for rendering.** Always use EGL (`MUJOCO_GL=egl`). OSMesa is
too slow for policy evaluation.

slurm_tools has its own git repo, use that to push changes to it.

## Architecture

### Training Pipeline

The system implements a RECAP-style RL pipeline:

1. **collect.py** — Roll out policy in LIBERO sim using LeRobot's
   `eval_policy()`, create LeRobot dataset with per-episode `success` labels.
1. **train_value.py** — Train distributional value function (SmolVLM + expert
   backbone, cross-entropy over discretized return bins).
1. **compute_advantage_labels.py** — Pre-compute binary advantage labels using
   the trained value model. Computes per-task advantage thresholds (30th
   percentile), binarizes per-sample advantages, and writes an `advantage_label`
   column directly into the dataset's parquet files.
1. **lerobot-train (advantage_train.yaml)** — Fine-tune SmolVLA with binary
   advantage conditioning via the `lerobot_policy_advantage` plugin. 30%
   advantage token dropout for classifier-free guidance.

### Advantage Policy Plugin (`lerobot_policy_advantage/`)

LeRobot plugin that registers the "advantage" policy type:

- **modeling_advantage.py** — `AdvantagePolicy`: wraps SmolVLA with text-based
  advantage conditioning ("Advantage: positive"/"negative").
- **configuration_advantage.py** — `AdvantageConfig` dataclass for policy
  configuration.
- **processor_advantage.py** — Delegates to SmolVLA's pre/post processors.

### Supporting Modules (`distal/`)

- **value_model.py** — `ValueModel` class: SmolVLM + expert with learnable value
  query token and categorical value head. Vision encoder frozen, VLM + expert
  trainable.
- **embedding.py** — VLM prefix extraction for PI05/SmolVLA. Mean-pooled
  embeddings over image tokens.
- **compute_maha_stats.py** — Computes Mahalanobis distance statistics (mean,
  covariance inverse) from embeddings using Ledoit-Wolf covariance.
- **rollout_value_viz.py** — Roll out base policy in LIBERO with value estimates
  and Rerun visualization.
- **visualize.py** — Rerun-based visualization of rollout traces synced with MP4
  videos.
- **push_to_hub.py** — Upload trained checkpoints to HuggingFace Hub.
- **trim_wandb_run.py** — Trim W&B runs by re-logging up to a given step.
- **hardware/zero.py** — Piper arm initialization/zeroing utility.
- **hardware/can_activate.py** — Activate and configure CAN interfaces for
  Piper.

### SLURM Tools (external git dependency)

Separate package (`slurm-tools`) pulled in as a git dependency. Provides cluster
job management:

- **slurm.py** — SLURM job submission via SSH (fabric). Builds sbatch scripts,
  rsyncs project to cluster, runs in Singularity containers. Also manages the
  GUI daemon (start/stop).
- **gui/app.py** — Flask web UI for monitoring SLURM jobs, GPU availability,
  streaming logs (SSE).

### Hardware Plugins

Two separate packages registered as LeRobot plugins:

- **lerobot_robot_piper/** — Piper arm as a LeRobot robot environment
  (PiperConfig, Piper class). Handles RealSense cameras with platform-specific
  variants (macOS/Linux).
- **lerobot_teleoperator_piper/** — Piper teleoperator interface for recording
  demonstrations.

### Configuration

YAML configs in `configs/` drive all workflows:

- `train.yaml` — Dataset, policy type, training hyperparameters, W&B logging
- `eval.yaml` — Evaluation settings (policy args via CLI override)
- `advantage_train.yaml` — Advantage-conditioned policy fine-tuning (policy
  repo, advantage dropout, eval settings)
- `slurm.yaml` — SLURM job submission settings (cluster paths, resources,
  container config)
- `sky.yaml` — SkyPilot cloud training config (cloud provider, accelerator,
  training task)
- `play.yaml` / `record.yaml` — Hardware interaction settings

### Deployment

Singularity container defined in `container.def` (MuJoCo EGL rendering). Targets
L40S/H100 GPUs on HTC cluster. Also supports cloud training via SkyPilot (Vast,
RunPod, etc.). Uses a custom LeRobot fork
(`reeceomahoney/lerobot@feat/combined-fixes`).
