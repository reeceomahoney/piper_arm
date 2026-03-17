# DistAL

**Dist**ance-based **A**dvantage **L**earning for robotic imitation learning. A RECAP-style RL pipeline for training VLAs with advantage-conditioned fine-tuning and Mahalanobis distance based rewards, built on top of [LeRobot](https://github.com/huggingface/lerobot).

## Pipeline

The system implements a multi-stage training pipeline:

1. **Collect rollouts** — Roll out and record a base VLA policy.

2. **Train value function** — Train a distributional value model using cross-entropy over discretized return bins. Supports both step-based and Mahalanobis distance-based reward signals.

3. **Compute advantage labels** — Use the trained value model to compute n-step TD advantages for every frame, then binarize them using per-task percentile thresholds.

4. **Fine-tune with advantage conditioning** — Fine-tune the VLA with binary advantage tokens.

## Project Structure

```
distal/                             # Core pipeline
├── collect.py                      # Policy rollout → dataset
├── train_value.py                  # Distributional value function training
├── compute_advantage_labels.py     # N-step TD advantages + binarization
├── value_model.py                  # ValueModel (SmolVLM + expert + categorical head)
├── embedding.py                    # VLM prefix extraction for OOD detection
├── mahalanobis.py                  # Mahalanobis distance + Gaussian fitting
├── rollout_value_viz.py            # Rerun visualization of rollouts + value estimates
└── visualize.py                    # Episode trace visualization

lerobot_policy_advantage/           # LeRobot plugin: advantage-conditioned policy
├── configuration_advantage.py      # AdvantageConfig
├── modeling_advantage.py           # AdvantagePolicy (SmolVLA + advantage embeddings)
└── processor_advantage.py          # Pre/post processors

lerobot_robot_piper/                # LeRobot plugin: Piper arm hardware interface
lerobot_teleoperator_piper/         # LeRobot plugin: Piper teleoperator for demos

configs/                            # YAML configs for all workflows
├── train.yaml                      # Base policy training
├── advantage_train.yaml            # Advantage-conditioned fine-tuning
├── eval.yaml                       # LIBERO evaluation
├── slurm.yaml                      # HPC cluster submission
├── record.yaml                     # Teleop recording
└── play.yaml                       # Hardware playback
```

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Usage

Workflows are driven by [mise](https://mise.jdx.dev/) tasks and YAML configs. Cluster job submission is managed with [slurm-tools](https://github.com/reeceomahoney/slurm-tools).

```bash
# Training & evaluation
mise run train              # Train base SmolVLA policy
mise run eval               # Evaluate in LIBERO simulation

# RL pipeline
uv run python -m distal.collect
uv run python -m distal.train_value
uv run python -m distal.compute_advantage_labels
mise run train              # with configs/advantage_train.yaml

# Hardware
mise run record             # Record demonstrations via teleop
mise run play               # Play trained policy on physical arm

# Cluster
mise run container          # Build Singularity container and upload to HPC
uv run slurm run            # Submit SLURM job
```

## Hardware

- **Piper robotic arm** — 6-DOF + gripper, controlled via CAN bus
- **2x Intel RealSense D435** — wrist-mounted and scene cameras (640x480 @ 30fps)
