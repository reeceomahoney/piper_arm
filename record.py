#!/usr/bin/env python3
"""
Record a LeRobot dataset using Piper arms in master-slave CAN configuration.

In this setup:
- The hardware handles teleoperation (master arm controls slave arm via CAN)
- Observations come from GetArmJointMsgs (actual joint positions)
- Actions come from GetArmJointCtrl (control commands being sent)

Usage:
    python record.py
"""

from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record

from piper_arm import PiperConfig, PiperTeleoperatorConfig


if __name__ == "__main__":
    # Robot configuration - reads observations from the follower arm
    robot_config = PiperConfig(
        can_interface="can0",
        use_degrees=True,
        include_gripper=True,
    )

    # Teleoperator configuration - reads actions from GetArmJointCtrl
    teleop_config = PiperTeleoperatorConfig(
        can_interface="can0",
        use_degrees=True,
        include_gripper=True,
        joint_names=robot_config.joint_names,
        joint_signs=robot_config.joint_signs,
    )

    # Dataset configuration
    dataset_config = DatasetRecordConfig(
        repo_id="reece-omahoney/piper_arm",
        single_task="Pick up the object",
        fps=10,
        episode_time_s=10,
        reset_time_s=1,
        num_episodes=10,
        video=True,
        push_to_hub=False,
        num_image_writer_processes=1,
        num_image_writer_threads_per_camera=2,
    )

    # Create the full recording config
    cfg = RecordConfig(
        robot=robot_config,
        teleop=teleop_config,
        dataset=dataset_config,
        display_data=False,
        play_sounds=True,
    )

    # Start recording
    record(cfg)
