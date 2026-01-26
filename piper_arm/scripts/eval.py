from pathlib import Path

from lerobot.configs.eval import EvalConfig, EvalPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.scripts.lerobot_eval import eval_main

PRETRAINED_PATH = "lerobot/pi05_libero_base"


def main():
    policy_cfg = PreTrainedConfig.from_pretrained(PRETRAINED_PATH)
    policy_cfg.pretrained_path = Path(PRETRAINED_PATH)
    policy_cfg.dtype = "bfloat16"

    cfg = EvalPipelineConfig(
        env=LiberoEnv(task="libero_object"),
        eval=EvalConfig(n_episodes=20, batch_size=1),
        policy=policy_cfg,
    )

    eval_main(cfg)


if __name__ == "__main__":
    main()
