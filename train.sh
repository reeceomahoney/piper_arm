lerobot-train \
  --dataset.repo_id=${HF_USER}/cube-pick-4 \
  --policy.type=act \
  --output_dir=outputs/train/act_cube_pick \
  --job_name=act_cube_pick \
  --policy.device=mps \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/my_policy
