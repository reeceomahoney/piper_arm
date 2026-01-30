from lerobot.configs.train import DatasetConfig, TrainPipelineConfig, WandBConfig
from lerobot.datasets.transforms import ImageTransformConfig, ImageTransformsConfig
from lerobot.envs.configs import AlohaEnv
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts.lerobot_train import train
from piper_arm.config import DATASET_NAME, EXP_NAME, HF_USER


def main():
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=f"{HF_USER}/{DATASET_NAME}",
            image_transforms=ImageTransformsConfig(
                enable=True,
                tfs={
                    "resize": ImageTransformConfig(
                        weight=1.0, type="Resize", kwargs={"size": (224, 224)}
                    )
                },
            ),
        ),
        # env=AlohaEnv(task="AlohaTransferCube-v0"),
        policy=ACTConfig(repo_id=f"{HF_USER}/{EXP_NAME}"),
        job_name=EXP_NAME,
        wandb=WandBConfig(enable=True),
        num_workers=8,
        batch_size=32,
        steps=250_000,
    )

    train(cfg)


if __name__ == "__main__":
    main()
