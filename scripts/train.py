from pathlib import Path

from lerobot.configs.train import DatasetConfig, TrainPipelineConfig, WandBConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts.lerobot_train import train

HF_USER = "reece-omahoney"
EXP_NAME = "act-cube-pick"
DATASET_NAME = "cube-pick-4"


def main():
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=f"{HF_USER}/{DATASET_NAME}"),
        policy=ACTConfig(
            repo_id=f"{HF_USER}/{EXP_NAME}",
            device="cuda:0",
            chunk_size=40,
            n_action_steps=40,
        ),
        output_dir=Path("outputs/train/") / EXP_NAME,
        job_name=EXP_NAME,
        wandb=WandBConfig(enable=True),
    )

    train(cfg)


if __name__ == "__main__":
    main()
