from lerobot.configs.train import DatasetConfig, TrainPipelineConfig, WandBConfig
from lerobot.envs.configs import AlohaEnv
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts.lerobot_train import train

from piper_arm.config import DATASET_NAME, EXP_NAME, HF_USER


def main():
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=f"{HF_USER}/{DATASET_NAME}"),
        # env=AlohaEnv(task="AlohaTransferCube-v0"),
        policy=ACTConfig(repo_id=f"{HF_USER}/{EXP_NAME}", device="cuda"),
        job_name=EXP_NAME,
        wandb=WandBConfig(enable=True),
        steps=int(1e6),
    )

    train(cfg)


if __name__ == "__main__":
    main()
