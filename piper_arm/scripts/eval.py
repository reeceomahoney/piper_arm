import os
from pathlib import Path

from lerobot.configs.eval import EvalConfig, EvalPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import AlohaEnv
from lerobot.scripts.lerobot_eval import eval_main

import piper_arm.policies.configuration_act_resize  # noqa: F401

PRETRAINED_PATH = "reece-omahoney/act-aloha-transfer-cube-finetuned"

os.environ["MUJOCO_GL"] = "egl"

def main():
    policy_cfg = PreTrainedConfig.from_pretrained(PRETRAINED_PATH)
    policy_cfg.pretrained_path = Path(PRETRAINED_PATH)

    cfg = EvalPipelineConfig(
        env=AlohaEnv(task="AlohaTransferCube-v0"),
        eval=EvalConfig(n_episodes=500, batch_size=50),
        policy=policy_cfg,
    )

    eval_main(cfg)


if __name__ == "__main__":
    main()
