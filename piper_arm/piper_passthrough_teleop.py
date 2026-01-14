# Passthrough teleoperator for Piper arms in master-slave CAN configuration
#
# This teleoperator reads control commands (GetArmJointCtrl) from the CAN bus
# rather than sending commands. Use this when the hardware already handles
# master-slave teleoperation and you just need to record the actions.

import logging
from dataclasses import dataclass, field
from typing import Any

from lerobot.teleoperators import Teleoperator, TeleoperatorConfig

log = logging.getLogger(__name__)

try:
    from piper_sdk import C_PiperInterface_V2
except Exception:
    C_PiperInterface_V2 = None


@TeleoperatorConfig.register_subclass("piper_passthrough")
@dataclass
class PiperPassthroughConfig(TeleoperatorConfig):
    # CAN interface to read control commands from (typically the follower arm)
    can_interface: str = "can0"
    # Joint names matching the robot configuration
    joint_names: list[str] = field(
        default_factory=lambda: [f"joint_{i + 1}" for i in range(6)]
    )
    # Sign flips to match robot configuration
    joint_signs: list[int] = field(default_factory=lambda: [-1, 1, 1, -1, 1, -1])
    # Include gripper in actions
    include_gripper: bool = False
    # When True, actions are in degrees/mm; when False, normalized [-100,100]
    use_degrees: bool = True


class PiperPassthroughTeleoperator(Teleoperator):
    """
    A passthrough teleoperator for Piper arms in hardware master-slave configuration.

    This teleoperator reads the control commands (GetArmJointCtrl) that the master
    is sending, rather than generating new commands. The actual teleoperation is
    handled by the CAN hardware - this just records what commands are being sent.

    Unlike the full PiperSDKInterface, this uses a read-only connection that
    doesn't call EnablePiper, so it can coexist with another SDK interface
    on the same CAN bus.

    Use this with a Robot configured with passthrough_mode=True.
    """

    config_class = PiperPassthroughConfig
    name = "piper_passthrough"

    def __init__(self, config: PiperPassthroughConfig):
        super().__init__(config)
        self.config = config
        self._piper = None

    @property
    def action_features(self) -> dict:
        ft = {f"{name}.pos": float for name in self.config.joint_names}
        if self.config.include_gripper:
            ft["gripper.pos"] = float
        return ft

    @property
    def feedback_features(self) -> dict:
        # No feedback needed - hardware handles it
        return {}

    @property
    def is_connected(self) -> bool:
        return self._piper is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if C_PiperInterface_V2 is None:
            raise ImportError("piper_sdk is not installed. Install with `pip install piper_sdk`.")

        if self._piper is None:
            self._piper = C_PiperInterface_V2(self.config.can_interface)
            self._piper.ConnectPort()
            # NOTE: We intentionally skip EnablePiper() here since the robot
            # will handle that. We just need to read the control commands.
            log.info("PiperPassthroughTeleoperator connected (read-only mode)")

        self.configure()

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, Any]:
        """
        Read the control commands from GetArmJointCtrl.
        These are the commands the master arm is sending.
        """
        if not self.is_connected or self._piper is None:
            raise ConnectionError(f"{self} is not connected.")

        jc = self._piper.GetArmJointCtrl().joint_ctrl
        gc = self._piper.GetArmGripperCtrl()

        action = {}
        for i, name in enumerate(self.config.joint_names, start=1):
            raw_val = getattr(jc, f"joint_{i}") / 1000.0  # SDK uses millidegrees
            deg = raw_val * self.config.joint_signs[i - 1]
            action[f"{name}.pos"] = deg

        if self.config.include_gripper:
            try:
                action["gripper.pos"] = gc.gripper_ctrl.grippers_angle / 10000.0
            except Exception:
                action["gripper.pos"] = 0.0

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # No-op: hardware handles the actual control
        pass

    def disconnect(self) -> None:
        # Just clear the reference - we don't own the CAN connection
        self._piper = None
