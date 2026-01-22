from pathlib import Path

from lerobot.configs.train import PreTrainedConfig
from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record

from piper_arm import PiperConfig
from piper_arm.config import DATASET_NAME, EXP_NAME, HF_USER


def main():
    policy_cfg = PreTrainedConfig.from_pretrained(f"{HF_USER}/{EXP_NAME}")
    policy_cfg.pretrained_path = Path(f"{HF_USER}/{EXP_NAME}")

    cfg = RecordConfig(
        robot=PiperConfig(teleop_mode=False),
        dataset=DatasetRecordConfig(
            repo_id=f"{HF_USER}/eval_{DATASET_NAME}",
            single_task="Pick and place the cube",
            fps=20,
            episode_time_s=600,
            reset_time_s=10,
            num_episodes=50,
            video=True,
            push_to_hub=False,
        ),
        policy=policy_cfg,
    )

    record(cfg)


if __name__ == "__main__":
    main()
