#!/bin/bash
set -euo pipefail

sudo apt-get update && sudo apt-get install -y \
  cmake \
  ffmpeg \
  g++ \
  libegl1 \
  libexpat1 \
  libfontconfig1-dev \
  libgl1 \
  libglvnd0 \
  libmagickwand-dev \
  libopengl0 \
  unzip

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd ~/sky_workdir

uv sync

LIBERO_DIR="$HOME/sky_workdir/.venv/lib/python3.12/site-packages/libero/libero"
if [ ! -d "$LIBERO_DIR/assets" ]; then
  TMP_EXTRACT=$(mktemp -d -p "$LIBERO_DIR")
  ZIP_PATH="$TMP_EXTRACT/libero_assets.zip"
  curl -L -o "$ZIP_PATH" \
    "https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip"
  unzip -q "$ZIP_PATH" -d "$TMP_EXTRACT"
  ASSETS_SRC=$(find "$TMP_EXTRACT" -type d -name assets -print -quit)
  mv "$ASSETS_SRC" "$LIBERO_DIR/assets"
  rm -rf "$TMP_EXTRACT"
fi

mkdir -p ~/.libero
cat <<EOF > ~/.libero/config.yaml
assets: $HOME/sky_workdir/.venv/lib/python3.12/site-packages/libero/libero/./assets
bddl_files: $HOME/sky_workdir/.venv/lib/python3.12/site-packages/libero/libero/./bddl_files
benchmark_root: $HOME/sky_workdir/.venv/lib/python3.12/site-packages/libero/libero
datasets: $HOME/sky_workdir/.venv/lib/python3.12/site-packages/libero/libero/../datasets
init_states: $HOME/sky_workdir/.venv/lib/python3.12/site-packages/libero/libero/./init_files
EOF

cd ~/sky_workdir
uv run python ~/sky_workdir/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py
