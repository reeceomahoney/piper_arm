#!/usr/bin/env python3
"""
Record a LeRobot dataset using Piper arms in master-slave CAN configuration.

In this setup:
- The hardware handles teleoperation (master arm controls slave arm via CAN)
- Observations come from GetArmJointMsgs (actual joint positions)
- Actions come from GetArmJointCtrl (control commands being sent)
- The robot's send_action is a no-op (passthrough_mode=True)

Usage:
    python record.py
"""

from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record

from piper_arm import PiperConfig, PiperPassthroughConfig


if __name__ == "__main__":
    # Robot configuration - reads observations from the follower arm
    # passthrough_mode=True means send_action is a no-op
    robot_config = PiperConfig(
        can_interface="can0",
        passthrough_mode=True,  # Hardware handles control
        use_degrees=True,
        include_gripper=True,
    )

    # Teleoperator configuration - reads actions from GetArmJointCtrl
    # This reads the control commands the master is sending
    teleop_config = PiperPassthroughConfig(
        can_interface="can0",  # Same CAN interface as robot
        use_degrees=True,
        include_gripper=True,
        # Match the robot's joint configuration
        joint_names=robot_config.joint_names,
        joint_signs=robot_config.joint_signs,
    )

    # Dataset configuration
    dataset_config = DatasetRecordConfig(
        repo_id="reece-omahoney/piper_arm",
        single_task="Pick up the object",  # Change this
        fps=10,
        episode_time_s=10,
        reset_time_s=1,
        num_episodes=10,
        video=True,  # Set True if using cameras
        push_to_hub=False,  # Set True to upload to HuggingFace
        num_image_writer_processes=1,
        num_image_writer_threads_per_camera = 2,
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
