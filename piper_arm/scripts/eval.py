import os
from pathlib import Path

import torch
from lerobot.configs.eval import EvalConfig, EvalPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.scripts.lerobot_eval import eval_main

import piper_arm.policies.configuration_act_resize  # noqa: F401

DEVICE_ID = 0
PRETRAINED_PATH = "reece-omahoney/smolvla-libero"


def main():
    os.environ["MUJOCO_GL"] = "egl"
    if torch.cuda.device_count() > 1:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = "1" if DEVICE_ID == 0 else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

    policy_cfg = PreTrainedConfig.from_pretrained(PRETRAINED_PATH)
    policy_cfg.pretrained_path = Path(PRETRAINED_PATH)

    cfg = EvalPipelineConfig(
        env=LiberoEnv("libero_object"),
        eval=EvalConfig(n_episodes=1, batch_size=1),
        policy=policy_cfg,
    )

    eval_main(cfg)


if __name__ == "__main__":
    main()
