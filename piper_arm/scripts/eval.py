from lerobot.configs.eval import EvalConfig, EvalPipelineConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.policies import SmolVLAConfig
from lerobot.scripts.lerobot_eval import eval_main


def main():
    cfg = EvalPipelineConfig(
        env=LiberoEnv(task="libero_object"),
        eval=EvalConfig(n_episodes=1, batch_size=1),
        policy=SmolVLAConfig.from_pretrained(
            pretrained_name_or_path="HuggingFaceVLA/smolvla_libero"
        ),
    )

    eval_main(cfg)


if __name__ == "__main__":
    main()
