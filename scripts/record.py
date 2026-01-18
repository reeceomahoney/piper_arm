"""
Record a LeRobot dataset using Piper arms in master-slave CAN configuration.

In this setup:
- The hardware handles teleoperation (master arm controls slave arm via CAN)
- Observations come from GetArmJointMsgs (actual joint positions)
- Actions come from GetArmJointCtrl (control commands being sent)
"""

from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record

from piper_arm import PiperConfig, PiperTeleoperatorConfig


def main():
    cfg = RecordConfig(
        robot=PiperConfig(),
        dataset=DatasetRecordConfig(
            repo_id="reece-omahoney/piper_arm",
            single_task="Pick up the object",
            fps=10,
            episode_time_s=10,
            reset_time_s=1,
            num_episodes=10,
            video=True,
            push_to_hub=False,
        ),
        teleop=PiperTeleoperatorConfig(),
    )

    record(cfg)


if __name__ == "__main__":
    main()
