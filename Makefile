.PHONY: train train-vae finetune eval record play sync submit list monitor logs clean-logs container fix-metaworld

METAWORLD_CFG := $(shell uv run python -c "from pathlib import Path; import lerobot; print(Path(lerobot.__file__).parent / 'envs' / 'metaworld_config.json')")

fix-metaworld:
	@test -f $(METAWORLD_CFG) || { \
		echo "Patching missing metaworld_config.json..."; \
		curl -sfL -o $(METAWORLD_CFG) \
			https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/envs/metaworld_config.json; \
	}

############
# Training #
############

train: fix-metaworld
	uv run lerobot-train --config_path configs/train.yaml

train-vae:
	uv run piper_arm/train_vae.py

finetune:
	uv run lerobot-train --config_path configs/train.yaml \
		--policy.path=lerobot/smolvla_base \
		--policy.repo_id=reece-omahoney/foobar \
		--policy.device=cuda \
		--policy.n_action_steps=10

eval:
	uv run lerobot-eval --config_path configs/eval.yaml \
		--policy.path=reece-omahoney/smolvla-libero-256

############
# Hardware #
############

record:
	uv run lerobot-record --config_path configs/record.yaml

play:
	uv run lerobot-record --config_path configs/play.yaml \
		--policy.path=reece-omahoney/smolvla-libero-256

#########
# SLURM #
#########

REMOTE_HOST = htc
REMOTE_PATH = ${DATA}/piper_arm

sync:
	rsync -avz --filter=':- .gitignore' ./ $(REMOTE_HOST):$(REMOTE_PATH)

submit: sync
	ssh $(REMOTE_HOST) 'cd $(REMOTE_PATH) && sbatch bin/submit.sh'

list:
	@ssh $(REMOTE_HOST) 'sinfo -N -p short -o "%.20N %.5t %.15C %.10m %.20G" | awk "NR==1 || /h100/ || /l40s/"'

monitor:
	ssh $(REMOTE_HOST) 'squeue -u $$USER'

cancel:
	ssh $(REMOTE_HOST) 'scancel -u $$USER'

logs:
	@set -- $$(ssh $(REMOTE_HOST) 'squeue -u $$USER -h -t R -o "$(REMOTE_PATH)/slurm-%i.out"'); \
	first=$$1; shift; \
	for f; do \
		tmux split-window -v "ssh $(REMOTE_HOST) 'tail -f $$f'"; \
	done; \
	ssh $(REMOTE_HOST) "tail -f $$first"

clean-logs:
	ssh $(REMOTE_HOST) 'cd $(REMOTE_PATH) && running=$$(squeue -u $$USER -h -t R -o "slurm-%i.out"); \
		if [ -n "$$running" ]; then \
			ls slurm-*.out 2>/dev/null | grep -vF "$$running" | xargs -r rm -f; \
		else \
			rm -f slurm-*.out; \
		fi'

###############
# Singularity #
###############

container: sync
	ssh $(REMOTE_HOST) 'cd $(REMOTE_PATH) && sbatch bin/build.sh'
