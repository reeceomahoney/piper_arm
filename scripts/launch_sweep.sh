#!/usr/bin/env bash
# Create (or reuse) a W&B sweep and submit N SLURM agents to the HTC cluster.
#
# Each agent runs `wandb agent --count 1` inside the same singularity container
# used by `mise run train`, so it executes exactly one sweep trial and exits.
# Submit one agent = one HTC job = one sweep trial.
#
# Usage:
#   ./scripts/launch_sweep.sh                       # creates sweep, submits 5 agents
#   ./scripts/launch_sweep.sh 10                    # creates sweep, submits 10 agents
#   SWEEP_ID=abc123 ./scripts/launch_sweep.sh 5     # reuse existing sweep
#
# Requires WANDB_API_KEY and HF_TOKEN exported in the local shell — they are
# baked into the SLURM script as literals at submit time and read by the
# singularity --env flags on the HTC node.

set -euo pipefail

N_AGENTS=${1:-5}
ENTITY="reeceomahoney"
PROJECT="distal"
SWEEP_CONFIG="configs/sweep.yaml"

cd "$(dirname "$0")/.."

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY not set in local env" >&2
  exit 1
fi

if [[ -z "${SWEEP_ID:-}" ]]; then
  echo "Creating sweep from $SWEEP_CONFIG ..."
  # `wandb sweep` prints "Creating sweep with ID: xxxx" to stderr.
  SWEEP_OUT=$(uv run wandb sweep --entity "$ENTITY" --project "$PROJECT" "$SWEEP_CONFIG" 2>&1)
  echo "$SWEEP_OUT"
  SWEEP_ID=$(echo "$SWEEP_OUT" | grep -oE 'sweep with ID: [a-z0-9]+' | awk '{print $NF}')
  if [[ -z "$SWEEP_ID" ]]; then
    echo "ERROR: failed to parse sweep ID from wandb output" >&2
    exit 1
  fi
fi

SWEEP_PATH="${ENTITY}/${PROJECT}/${SWEEP_ID}"
echo "Sweep: https://wandb.ai/${SWEEP_PATH}"

# Single-line command for the SLURM script. Single-quoted so $WANDB_API_KEY /
# $HF_TOKEN stay as literal text — they get expanded on the HTC node where the
# singularity --env flags read them.
AGENT_CMD='singularity run --nv --bind /data/engs-robotics-ml/kebl6123/.cache:/data/engs-robotics-ml/kebl6123/.cache --env "WANDB_API_KEY=${WANDB_API_KEY}" --env "HF_TOKEN=${HF_TOKEN}" --env "PYTHONPATH=/data/engs-robotics-ml/kebl6123/distal:/data/engs-robotics-ml/kebl6123/distal/lerobot_policy_pistar06" container.sif wandb agent --count 1 '"$SWEEP_PATH"

echo
echo "Submitting $N_AGENTS SLURM agent(s) for $SWEEP_PATH ..."
for i in $(seq 1 "$N_AGENTS"); do
  echo "  → agent $i/$N_AGENTS"
  uv run slurm run --command "$AGENT_CMD" --time 10
done

echo
echo "Done. Monitor at: https://wandb.ai/${SWEEP_PATH}"
echo "To submit more agents later: SWEEP_ID=$SWEEP_ID ./scripts/launch_sweep.sh <N>"
