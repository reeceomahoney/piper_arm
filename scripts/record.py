"""
Record a LeRobot dataset using Piper arms in master-slave CAN configuration.

In this setup:
- The hardware handles teleoperation (master arm controls slave arm via CAN)
- Observations come from GetArmJointMsgs (actual joint positions)
- Actions come from GetArmJointCtrl (control commands being sent)
"""

from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record

from piper_arm import PiperConfig, PiperTeleoperatorConfig
from .train import HF_USER


def main():
    cfg = RecordConfig(
        robot=PiperConfig(),
        dataset=DatasetRecordConfig(
            repo_id=f"{HF_USER}/piper-arm",
            single_task="Pick up the object",
            fps=20,
            episode_time_s=600,
            reset_time_s=10,
            num_episodes=50,
            video=True,
            push_to_hub=False,
        ),
        teleop=PiperTeleoperatorConfig(),
        display_data=True,
    )

    record(cfg)


if __name__ == "__main__":
    main()
