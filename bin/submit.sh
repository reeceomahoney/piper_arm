#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --gres=gpu:h100:1

set -euo pipefail

echo "Starting at $(date +%H:%M)"

singularity run \
  --nv --env "WANDB_API_KEY=${WANDB_API_KEY}" --env "HF_TOKEN=${HF_TOKEN}" \
  container.sif make train
