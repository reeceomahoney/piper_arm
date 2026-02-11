from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("piper")
@dataclass
class PiperConfig(RobotConfig):
    can_interface: str = "can0"
    joint_names: list[str] = field(
        default_factory=lambda: [f"joint_{i + 1}" for i in range(6)]
    )
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "wrist": RealSenseCameraConfig(
                serial_number_or_name="123622270993", fps=30, width=640, height=480
            ),
            "scene": RealSenseCameraConfig(
                serial_number_or_name="128422270436", fps=30, width=640, height=480
            ),
        }
    )
    teleop_mode: bool = True
