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
