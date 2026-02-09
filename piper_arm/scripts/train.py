import os

import torch
from lerobot.configs.eval import EvalConfig
from lerobot.configs.train import DatasetConfig, TrainPipelineConfig, WandBConfig
from lerobot.envs.configs import AlohaEnv, LiberoEnv
from lerobot.policies.factory import SmolVLAConfig
from lerobot.scripts.lerobot_train import train

# import piper_arm.envs  # noqa: F401
from piper_arm.config import DATASET_NAME, EXP_NAME, HF_USER

DEVICE_ID = 0
PRETRAINED_PATH = None


def main():
    if torch.cuda.device_count() > 1:
        # os.environ["MUJOCO_EGL_DEVICE_ID"] = "1" if DEVICE_ID == 0 else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=f"{DATASET_NAME}"),
        # env=AlohaEnv(task="AlohaTransferCubeWithTask-v0"),
        env=LiberoEnv("libero_object"),
        policy=SmolVLAConfig(
            repo_id=f"{HF_USER}/{EXP_NAME}",
            n_action_steps=10,
            load_vlm_weights=True,
        ),
        job_name=EXP_NAME,
        eval=EvalConfig(n_episodes=1, batch_size=1),
        wandb=WandBConfig(enable=True),
        num_workers=8,
        batch_size=64,
        steps=200_000,
        eval_freq=5_000,
    )

    train(cfg)


if __name__ == "__main__":
    main()
