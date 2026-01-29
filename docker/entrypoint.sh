#!/bin/bash
set -e

uv sync --link-mode=copy
uv run wandb login $(cat /run/secrets/WANDB_API_KEY)
export HF_TOKEN=$(cat /run/secrets/HF_TOKEN)

exec "$@"
