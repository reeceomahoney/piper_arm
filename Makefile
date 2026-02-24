.PHONY: train train-vae finetune eval eval-dist record play fetch-eval gui

############
# Training #
############

train:
	lerobot-train --config_path configs/train.yaml

train-vae:
	uv run piper_arm/train_vae.py

finetune:
	uv run lerobot-train --config_path configs/train.yaml \
		--policy.path=lerobot/smolvla_base \
		--policy.repo_id=reece-omahoney/foobar \
		--policy.device=cuda \
		--policy.n_action_steps=10

eval:
	lerobot-eval --config_path configs/eval.yaml \
		--policy.path=lerobot/pi05_libero_finetuned

eval-dist:
	python piper_arm/eval_dist.py

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

fetch-eval:
	rsync -avz $(REMOTE_HOST):$(REMOTE_PATH)/outputs/eval_dist/ ./outputs/eval_dist/

gui:
	uv run --extra gui piper_arm/gui/app.py
