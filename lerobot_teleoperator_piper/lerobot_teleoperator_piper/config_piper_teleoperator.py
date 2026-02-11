from dataclasses import dataclass, field

from lerobot.teleoperators import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("piper_teleop")
@dataclass
class PiperTeleoperatorConfig(TeleoperatorConfig):
    can_interface: str = "can0"
    joint_names: list[str] = field(
        default_factory=lambda: [f"joint_{i + 1}" for i in range(6)]
    )
