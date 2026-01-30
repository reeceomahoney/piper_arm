from lerobot.configs.train import DatasetConfig, TrainPipelineConfig, WandBConfig
from lerobot.envs.configs import AlohaEnv
from lerobot.scripts.lerobot_train import train

from piper_arm.config import DATASET_NAME, EXP_NAME, HF_USER
from piper_arm.policies.configuration_act_resize import ACTResizeConfig


def main():
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=f"{HF_USER}/{DATASET_NAME}"),
        # env=AlohaEnv(task="AlohaTransferCube-v0"),
        policy=ACTResizeConfig(repo_id=f"{HF_USER}/{EXP_NAME}", resize_size=(224, 224)),
        job_name=EXP_NAME,
        wandb=WandBConfig(enable=True),
        num_workers=8,
        batch_size=32,
        steps=250_000,
    )

    train(cfg)


if __name__ == "__main__":
    main()
