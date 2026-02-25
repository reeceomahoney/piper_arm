.PHONY: train finetune eval record play gui gui-start gui-stop container

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

#######
# GUI #
#######

gui-start:
	@if [ -f /tmp/piper-gui.pid ] && kill -0 $$(cat /tmp/piper-gui.pid) 2>/dev/null; then \
		echo "GUI already running (PID $$(cat /tmp/piper-gui.pid))"; \
	else \
		rm -f /tmp/piper-gui.pid; \
		nohup uv run --extra gui piper_arm/gui/app.py &> /tmp/piper-gui.log & echo $$! > /tmp/piper-gui.pid; \
		echo "GUI started (PID $$(cat /tmp/piper-gui.pid))"; \
	fi

gui-stop:
	@kill $$(cat /tmp/piper-gui.pid) 2>/dev/null && rm -f /tmp/piper-gui.pid && echo "GUI stopped" || echo "GUI not running"

#############
# Container #
#############

REMOTE_HOST := htc
REMOTE_PATH := /data/engs-robotics-ml/kebl6123/piper_arm

container:
	singularity build --fakeroot container.sif container.def
	scp container.sif $(REMOTE_HOST):$(REMOTE_PATH)/container.sif
