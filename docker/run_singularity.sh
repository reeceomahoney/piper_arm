#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE="${SINGULARITY_IMAGE:-docker://reeceomahoney/piper-arm:latest}"

singularity run \
  --nv \
  --env "TERM=${TERM},PYTHONDONTWRITEBYTECODE=1,WANDB_API_KEY=${WANDB_API_KEY},HF_TOKEN=${HF_TOKEN}" \
  --bind "${PROJECT_DIR}/piper_arm:/work/piper_arm" \
  --bind "${PROJECT_DIR}/outputs:/work/outputs" \
  "$IMAGE" \
  uv run train
