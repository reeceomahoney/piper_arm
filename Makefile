.PHONY: sync submit list monitor logs clean-logs push

REMOTE_HOST = htc
REMOTE_PATH = ${DATA}/piper_arm

sync:
	rsync -avz --filter=':- .gitignore' ./ $(REMOTE_HOST):$(REMOTE_PATH)

submit: sync
	ssh $(REMOTE_HOST) 'cd $(REMOTE_PATH) && sbatch bin/submit.sh'

list:
	@ssh $(REMOTE_HOST) 'sinfo -N -p short -o "%.20N %.5t %.15C %.10m %.20G %.10A" | awk "NR==1 || /h100/"'

monitor:
	ssh $(REMOTE_HOST) 'squeue -u $$USER'

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

push:
	docker build -t reeceomahoney/piper-arm:latest -f docker/Dockerfile .
	docker push reeceomahoney/piper-arm:latest

train:
	uv run lerobot-train --config_path configs/train.yaml
