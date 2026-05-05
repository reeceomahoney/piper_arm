#!/bin/bash
set -euo pipefail

sudo apt-get update && sudo apt-get install -y ffmpeg

curl https://mise.run | sh
echo 'eval "$($HOME/.local/bin/mise activate bash)"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
eval "$(mise activate bash)"
cd ~/sky_workdir
mise trust

uv sync
