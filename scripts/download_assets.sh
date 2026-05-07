#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/slurm-%j.out

# Populate the host-side libero cache so the Singularity container can find
# LIBERO-plus assets without baking them into the image. Submit once on HTC
# via `pixi run download-assets`.
set -euo pipefail

CACHE_DIR="${CACHE_DIR:-/data/engs-robotics-ml/kebl6123/.cache}"
LIBERO_CACHE="$CACHE_DIR/libero"
ASSETS_DIR="$LIBERO_CACHE/assets"

# Paths inside the container — libero reads config.yaml from inside, so these
# must match the venv layout in container.def.
VENV_LIBERO="/opt/distal/.venv/lib/python3.12/site-packages/libero/libero"

mkdir -p "$LIBERO_CACHE"

if [ ! -d "$ASSETS_DIR" ]; then
    TMP_EXTRACT=$(mktemp -d -p "$LIBERO_CACHE")
    ZIP_PATH="$TMP_EXTRACT/libero_assets.zip"
    curl -L -o "$ZIP_PATH" \
        "https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip"
    unzip -q "$ZIP_PATH" -d "$TMP_EXTRACT"
    ASSETS_SRC=$(find "$TMP_EXTRACT" -type d -name assets -print -quit)
    mv "$ASSETS_SRC" "$ASSETS_DIR"
    rm -rf "$TMP_EXTRACT"
fi

cat > "$LIBERO_CACHE/config.yaml" <<EOF
assets: $ASSETS_DIR
bddl_files: $VENV_LIBERO/bddl_files
benchmark_root: $VENV_LIBERO
datasets: $VENV_LIBERO/../datasets
init_states: $VENV_LIBERO/init_files
EOF

echo "libero cache ready at $LIBERO_CACHE"
