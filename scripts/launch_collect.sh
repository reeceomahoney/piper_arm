#!/usr/bin/env bash
# Fan out LIBERO-plus collection across one sky cluster per suite.
#
# Each suite runs on its own Vast cluster via configs/sky.yaml; clusters
# autostop+down after 10 idle minutes (see sky.yaml). Each shard is pushed to
# HF Hub as `${DATASET_REPO_ID}-${short}`; merge them afterwards with
# `lerobot-edit-dataset --operation.type merge`.
#
# Usage:
#   ./scripts/launch_collect.sh
#   DATASET_REPO_ID=reece-omahoney/pi05-libero-plus ./scripts/launch_collect.sh
#
# Requires HF_TOKEN and WANDB_API_KEY exported in the local shell.

set -euo pipefail

cd "$(dirname "$0")/.."

BASE_REPO_ID="${DATASET_REPO_ID:-reece-omahoney/pi05-libero-plus}"
SUITES=(libero_spatial libero_object libero_goal libero_10)

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set in local env" >&2
  exit 1
fi

pids=()
shard_clusters=()

# Stagger launches: SkyPilot's Vast provider races on VastAI client init when
# launches fire concurrently (one wins, others crash with "missing api_key").
# 15s between launches lets each finish provider init before the next starts;
# provisioning itself still proceeds in parallel.
STAGGER_SECS=15

for i in "${!SUITES[@]}"; do
  suite="${SUITES[$i]}"
  short="${suite#libero_}"
  shard="${BASE_REPO_ID}-${short}"
  cluster="distal-collect-${short}"
  shard_clusters+=("$cluster")
  echo "Launching $suite on cluster $cluster -> $shard"
  sky launch configs/sky.yaml -y \
    -c "$cluster" \
    --env HF_TOKEN --env WANDB_API_KEY \
    --env "SUITE=$suite" \
    --env "DATASET_REPO_ID=$shard" &
  pids+=($!)
  if (( i < ${#SUITES[@]} - 1 )); then
    sleep "$STAGGER_SECS"
  fi
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "ERROR: collection job for ${SUITES[$i]} (cluster ${shard_clusters[$i]}) failed" >&2
    fail=1
  fi
done

exit "$fail"
