"""
Record a LeRobot dataset using Piper arms in master-slave CAN configuration.

In this setup:
- The hardware handles teleoperation (master arm controls slave arm via CAN)
- Observations come from GetArmJointMsgs (actual joint positions)
- Actions come from GetArmJointCtrl (control commands being sent)
"""

from pathlib import Path

from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record

from piper_arm import PiperConfig, PiperTeleoperatorConfig


def main(root: Path | None = None):
    robot_config = PiperConfig(
        can_interface="can0",
        use_degrees=True,
        include_gripper=True,
    )

    teleop_config = PiperTeleoperatorConfig(
        can_interface="can0",
        use_degrees=True,
        include_gripper=True,
        joint_names=robot_config.joint_names,
        joint_signs=robot_config.joint_signs,
    )

    dataset_config = DatasetRecordConfig(
        repo_id="reece-omahoney/cube-pick-4",
        single_task="Cube Pick",
        root=root,
        fps=20,
        episode_time_s=60,
        reset_time_s=10,
        num_episodes=50,
        video=True,
        push_to_hub=False,
    )

    cfg = RecordConfig(
        robot=robot_config,
        teleop=teleop_config,
        dataset=dataset_config,
        display_data=True,
    )

    record(cfg)


if __name__ == "__main__":
    main()
