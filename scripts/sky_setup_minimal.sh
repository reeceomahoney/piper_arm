#!/bin/bash
set -euo pipefail

sudo apt-get update && sudo apt-get install -y ffmpeg

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd ~/sky_workdir

uv sync
