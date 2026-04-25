# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

DistAL: a RECAP-style RL pipeline for fine-tuning VLAs (Pi0.5 / SmolVLA) with
advantage conditioning and Mahalanobis-distance-based rewards, built on a fork
of HuggingFace LeRobot. Primary evaluation target is LIBERO simulation; also
supports a physical Piper arm.

**Python 3.12** (`>=3.12,<3.13`). **uv** package manager (frozen `uv.lock`),
**mise** task runner.

## Common Commands

```bash
# Base policy training & evaluation
mise run train                           # lerobot-train using configs/train.yaml
mise run eval                            # lerobot-eval in LIBERO sim (pi05-libero default)

# RECAP pipeline (run directly via uv, not lerobot-train)
uv run python -m distal.collect          # rollouts → LeRobot dataset
uv run python -m distal.collect_libero_plus
uv run python -m distal.compute_maha_stats      # mean / cov_inv from base-dataset embeddings
uv run python -m distal.train_value             # distributional value network
uv run python -m distal.train_pi_star           # advantage-conditioned Pi0.5 fine-tune
uv run python -m distal.maha_auroc              # Mahalanobis distance AUROC vs episode success
uv run python -m distal.eval_guidance           # sweep guidance scales

# Hardware (Piper)
mise run record                          # teleop demos
mise run play                            # play trained policy on the arm

# Cluster / cloud
mise run sky [cluster_id]                # launch on Vast via SkyPilot, or sky exec on existing
mise run sky-ssh [cluster_id]            # same but via SSH cloud
mise run container                       # build container.sif and scp to HTC
uv run slurm run                         # SLURM submit (slurm-tools git dep)
uv run slurm gui [stop]                  # Flask job-monitor daemon

# Quality
uv run pre-commit run --all-files        # ruff (E,F,I + format), check-toml/yaml, mdformat --wrap 80, ty
```

No formal test suite — verification is via `mise run eval` and the `maha_auroc`
diagnostic.

## Conventions

- **Never start function or variable names with underscores.** Use plain names.
- **Don't add `Usage:` sections to module docstrings** — entry points use
  `draccus`/`lerobot.configs.parser`, which are self-documenting.
- **Never use OSMesa for MuJoCo rendering. Always EGL** (`MUJOCO_GL=egl`,
  `PYOPENGL_PLATFORM=egl`). OSMesa is too slow for policy evaluation.
- `slurm-tools` is a separate git repo (pulled as a git dependency); push
  changes to it from its own checkout.
- **SkyPilot API server caches code.** After patching SkyPilot source under
  `.venv/`, run `uv run sky api stop` before retrying — the daemon keeps stale
  modules loaded.
- **PRs to external repos** (LeRobot fork etc.): check
  `.github/pull_request_template.md` and `CONTRIBUTING.md` first and follow
  their format.

## Architecture

### Pipeline

The system is a multi-stage pipeline; each stage produces an artifact consumed
by the next.

1. **Collect** (`distal/collect.py`, `distal/collect_libero_plus.py`) — Roll out
   a base policy in LIBERO via LeRobot's `eval_policy()`, save observations,
   actions, and per-episode `success` into a LeRobot dataset.
1. **Maha stats** (`distal/compute_maha_stats.py`) — From the base dataset the
   policy was trained on, fit Ledoit-Wolf mean / inverse covariance over
   mean-pooled VLM image-token embeddings (`distal/embedding.py`). Saved as
   safetensors and cached on the HF Hub.
1. **Train value** (`distal/train_value.py`) — Distributional value model
   (`RECAPValueNetwork` in `distal/value_model.py`: SmolVLM + expert + learned
   value query token + categorical head, vision encoder frozen). Reward signal
   is either fixed `-1` per step or `distal/maha_reward.py` (Mahalanobis-based
   `[-1, 0]` rewards). Adapted from the upstream LeRobot
   `jv/recap-value-network` PR.
1. **Train PiStar06** (`distal/train_pi_star.py`) — Advantage-conditioned Pi0.5
   fine-tune. **Advantages are pre-computed in this script** by running the
   frozen value network once over the dataset, then injected into batches via a
   frame-index → advantage dict. Caching is content-addressed by
   `distal/advantage_cache.py` (key = dataset + VN commit SHAs +
   hyperparameters), with cache files mirrored to a HF Hub `dataset` repo. There
   is no separate `compute_advantage_labels` step — it lives inside this script.

### PiStar06 Plugin (`lerobot_policy_pistar06/`)

LeRobot plugin registering the **`pistar06`** policy type. PiStar06 = Pi0.5
(`PI05Config`) extended with binary advantage conditioning injected via
text/embedding into the action expert (`embed_suffix`). Built with flat
`nn.Module` composition rather than the deep PaliGemma inheritance chain to
avoid ~3× peak memory during init. Key config knobs: `value_network_checkpoint`,
`enable_advantage_conditioning` (master switch, persisted in `config.json` so
inference matches training), `advantage_threshold` (resolved scalar, typically
auto-set to a per-task percentile during training), `advantage_dropout` (CFG).

### Supporting modules (`distal/`)

- `value_model.py` — `RECAPValueNetwork` (SmolVLM + expert backbone, value query
  token, categorical head over discretized return bins).
- `embedding.py` — Mean-pooled VLM prefix embeddings for both PI05 and SmolVLA.
- `maha_reward.py` — Loads stats from `compute_maha_stats.py`, computes per-
  frame Mahalanobis distances on a value-training dataset, min-max normalizes to
  `[-1, 0]` for use as per-step rewards.
- `maha_auroc.py` — Evaluates Mahalanobis distance as a failure predictor:
  per-frame distances → episode-mean → AUROC vs `success` labels.
- `advantage_cache.py` — Content-addressed cache for precomputed advantages,
  Hub-mirrored.
- `eval_guidance.py` — Sweeps classifier-free guidance scales by shelling out to
  `lerobot-eval`.
- `rollout_value_viz.py` — Rollout + Rerun visualization with value estimates.
- `push_to_hub.py` — Upload checkpoints / value networks to HF Hub.
- `plotting/` — Diagnostic scripts: `debug_maha_rewards.py`,
  `plot_gt_returns.py`. `plot_gt_returns.py` mirrors the exact reward/return
  construction in `train_value._build_frame_targets` so the plot reflects what
  the model actually trains against.
- `hardware/zero.py`, `hardware/can_activate.py` — Piper init / CAN bring-up.

### Hardware Plugins

- `lerobot_robot_piper/` — Piper arm (6-DOF + gripper, CAN bus) + 2× Intel
  RealSense D435 (wrist + scene, 640×480 @ 30fps). Platform-specific RealSense
  variants for macOS vs Linux.
- `lerobot_teleoperator_piper/` — Piper teleop interface for `mise run record`.

Both packages are commented out from the `dev` group in `pyproject.toml`; sync
locally only when working on hardware.

### Configs (`configs/`)

YAML configs drive workflows via draccus / LeRobot config parsers:

- `train.yaml` — base SmolVLA training (`mise run train`).
- `eval.yaml` — LIBERO eval; **policy args must come from CLI**, e.g.
  `mise run eval` overrides `--policy.path` and `--policy.n_action_steps`.
- `sky.yaml` / `sky-ssh.yaml` — SkyPilot launch configs (Vast / generic SSH),
  including LIBERO-plus assets bootstrap.
- `slurm.yaml` — HTC SLURM submission with Singularity bind mounts.
- `record.yaml` / `play.yaml` — Hardware workflows.

### Deployment

- **Singularity** (`container.def` → `container.sif`) for the HTC cluster
  (L40S/H100). Built and uploaded via `mise run container`.
- **SkyPilot** (`configs/sky*.yaml`) targets Vast / RunPod / etc. The setup
  block bootstraps `mise`, `uv sync`, and downloads LIBERO assets from the
  `Sylvest/LIBERO-plus` HF dataset.

### LeRobot fork

`pyproject.toml` pins `lerobot` to a custom fork:
`reeceomahoney/lerobot @ distal-libero-plus`. A `distal` branch is also
available (commented out). Switching branches requires a `uv lock` + `uv sync`.
