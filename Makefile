.PHONY: sync submit monitor logs clean-logs push

REMOTE_HOST = htc
REMOTE_PATH = ${DATA}/piper_arm

sync:
	rsync -avz --filter=':- .gitignore' ./ $(REMOTE_HOST):$(REMOTE_PATH)

submit: sync
	ssh $(REMOTE_HOST) 'cd $(REMOTE_PATH) && sbatch bin/submit.sh'

monitor:
	ssh $(REMOTE_HOST) 'squeue -u $$USER'

logs:
	ssh $(REMOTE_HOST) 'tail -f $$(squeue -u $$USER -h -t R -o "%i" | xargs -I {} echo $(REMOTE_PATH)/slurm-{}.out)'

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
