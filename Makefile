.PHONY: train finetune eval record play gui

############
# Training #
############

train:
	lerobot-train --config_path configs/train.yaml

finetune:
	lerobot-train --config_path configs/train.yaml \
		--policy.path=lerobot/smolvla_base \
		--policy.repo_id=reece-omahoney/foobar \
		--policy.device=cuda \
		--policy.n_action_steps=10

eval:
	lerobot-eval --config_path configs/eval.yaml \
		--policy.path=lerobot/pi05_libero_finetuned

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

gui:
	uv run --extra gui piper_arm/gui/app.py
