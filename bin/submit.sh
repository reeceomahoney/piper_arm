#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --gres=gpu:h100:1

set -euo pipefail

echo "Starting at $(date +%H:%M)"

PROJECT_DIR="$(pwd)"
IMAGE=docker://reeceomahoney/piper-arm:latest

singularity run \
  --nv \
  --env "TERM=${TERM}" \
  --env "PYTHONDONTWRITEBYTECODE=1" \
  --env "MUJOCO_GL=osmesa" \
  --env "WANDB_API_KEY=${WANDB_API_KEY}" \
  --env "HF_TOKEN=${HF_TOKEN}" \
  --env "UV_CACHE_DIR=/work/.cache/uv" \
  --env "HF_HOME=/work/.cache/huggingface" \
  --env "LIBERO_CONFIG_PATH=/work/.cache/libero" \
  --env "WANDB_CACHE_DIR=/work/.cache/wandb" \
  --env "UV_LINK_MODE=copy" \
  --bind "${PROJECT_DIR}/piper_arm:/work/piper_arm" \
  --bind "${PROJECT_DIR}/outputs:/work/outputs" \
  --bind "${PROJECT_DIR}/pyproject.toml:/work/pyproject.toml" \
  --bind "${PROJECT_DIR}/uv.lock:/work/uv.lock" \
  --bind "${PROJECT_DIR}/../.cache:/work/.cache" \
  "$IMAGE" \
  make train
