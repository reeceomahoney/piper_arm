.PHONY: train finetune eval train-advantage record play container

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
		--policy.path=reece-omahoney/smolvla-libero-16-chunk

train-advantage:
	lerobot-train --config_path configs/advantage_train.yaml

############
# Hardware #
############

record:
	lerobot-record --config_path configs/record.yaml

play:
	lerobot-record --config_path configs/play.yaml \
		--policy.path=reece-omahoney/smolvla-libero-256

#############
# Container #
#############

REMOTE_HOST := htc
REMOTE_PATH := /data/engs-robotics-ml/kebl6123/piper_arm

container:
	singularity build --force --fakeroot container.sif container.def
	scp container.sif $(REMOTE_HOST):$(REMOTE_PATH)/container.sif
