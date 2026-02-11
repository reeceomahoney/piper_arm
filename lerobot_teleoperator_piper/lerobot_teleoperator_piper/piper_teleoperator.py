import time
from typing import Any

from lerobot.teleoperators import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from piper_sdk import C_PiperInterface_V2

from .config_piper_teleoperator import PiperTeleoperatorConfig


class PiperTeleoperator(Teleoperator):
    config_class = PiperTeleoperatorConfig
    name = "piper_teleop"

    def __init__(self, config: PiperTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self.piper = C_PiperInterface_V2(self.config.can_interface)
        self._is_piper_connected = False

    @property
    def action_features(self) -> dict:
        ft = {f"{name}.pos": float for name in self.config.joint_names}
        ft["gripper.pos"] = float
        return ft

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_piper_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.piper.ConnectPort()
        time.sleep(0.1)
        self._is_piper_connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, Any]:
        jc = self.piper.GetArmJointCtrl().joint_ctrl
        gc = self.piper.GetArmGripperCtrl()

        action = {
            f"joint_{i}.pos": getattr(jc, f"joint_{i}") / 1000.0 for i in range(1, 7)
        }
        action["gripper.pos"] = gc.gripper_ctrl.grippers_angle / 10000.0

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        self.piper.DisconnectPort()
