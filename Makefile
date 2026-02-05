.PHONY: sync submit monitor logs shell clean-remote

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

shell:
	ssh $(REMOTE_HOST)

clean-remote:
	ssh $(REMOTE_HOST) 'cd $(REMOTE_PATH) && rm -f slurm-*.out'
