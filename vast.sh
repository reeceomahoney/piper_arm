set -euo pipefail

# bashrc (fill these in)
cat >> ~/.bashrc << 'EOF'
export PYTHONDONTWRITEBYTECODE=1
export HF_TOKEN=""
export WANDB_API_KEY=""
EOF
source ~/.bashrc

# dotfiles
git clone git@github.com:reeceomahoney/.dotfiles.git ~/.dotfiles
cd ~/.dotfiles
sudo apt install stow
stow .
tmux source-file ~/.config/tmux/tmux.conf

# project
cd /workspace
git clone git@github.com:reeceomahoney/distal.git
cd distal

# mise
curl https://mise.run | sh
echo "eval \"\$(/root/.local/bin/mise activate bash)\"" >> ~/.bashrc
source ~/.bashrc
cd /workspace/distal
mise trust

# uv
uv sync

# preflight: reject instances without NVIDIA EGL
python3 -c "
import os; os.environ['PYOPENGL_PLATFORM']='egl'
from OpenGL import EGL
d = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
assert EGL.eglInitialize(d, None, None), 'EGL init failed'
v = EGL.eglQueryString(d, EGL.EGL_VENDOR)
assert b'NVIDIA' in v, f'EGL vendor is {v}, not NVIDIA — no hardware EGL on this host'
print(f'EGL OK: {v}')
"

load=$(awk '{print $1}' /proc/loadavg)
echo "Load average: $load"
awk '{if ($1 > 2.0) { print "WARNING: High load, likely noisy neighbor. Destroy this instance."; exit 1 }}' /proc/loadavg
