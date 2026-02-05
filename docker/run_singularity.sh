#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE="${SINGULARITY_IMAGE:-docker://reeceomahoney/piper-arm:latest}"

singularity run \
  --nv \
  --env "TERM=${TERM}" \
  --env "PYTHONDONTWRITEBYTECODE=1" \
  --env "WANDB_API_KEY=${WANDB_API_KEY}" \
  --env "HF_TOKEN=${HF_TOKEN}" \
  --env "UV_CACHE_DIR=/work/.cache/uv" \
  --env "HF_HOME=/work/.cache/huggingface" \
  --bind "${PROJECT_DIR}/piper_arm:/work/piper_arm" \
  --bind "${PROJECT_DIR}/outputs:/work/outputs" \
  --bind "${PROJECT_DIR}/pyproject.toml:/work/pyproject.toml" \
  --bind "${PROJECT_DIR}/uv.lock:/work/uv.lock" \
  --bind "${PROJECT_DIR}/.cache:/work/.cache" \
  "$IMAGE" \
  uv run train
