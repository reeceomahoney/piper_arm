import time
from dataclasses import dataclass, field
from typing import Any

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots import Robot, RobotConfig
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from piper_sdk import C_PiperInterface_V2


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
            )
        }
    )


class Piper(Robot):
    config_class = PiperConfig
    name = "piper"

    def __init__(self, config: PiperConfig):
        super().__init__(config)
        self.config = config
        self.piper = C_PiperInterface_V2(self.config.can_interface)
        self.cameras = make_cameras_from_configs(config.cameras)
        self._is_piper_connected = False

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{j}.pos": float for j in self.config.joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {k: (c.height, c.width, 3) for k, c in self.cameras.items()}

    @property
    def observation_features(self) -> dict:
        ft = {**self._motors_ft, **self._cameras_ft}
        ft["gripper.pos"] = float
        return ft

    @property
    def action_features(self) -> dict:
        ft = {f"{name}.pos": float for name in self.config.joint_names}
        ft["gripper.pos"] = float
        return ft

    @property
    def is_connected(self) -> bool:
        return self._is_piper_connected and all(
            cam.is_connected for cam in self.cameras.values()
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.piper.ConnectPort()
        time.sleep(0.1)

        while not self.piper.EnablePiper():
            time.sleep(0.01)

        self._is_piper_connected = True

        # Reset
        self.piper.MotionCtrl_1(0x02, 0, 0)

        # SDK motor list appears 1-indexed -> extract 1..6
        # Angle limits are in deci-degrees (0.1 deg). Convert to degrees for consistency.
        limits = self.piper.GetAllMotorAngleLimitMaxSpd()
        self.min_pos = [
            pos.min_angle_limit / 10.0
            for pos in limits.all_motor_angle_limit_max_spd.motor[1:7]
        ] + [0.0]
        self.max_pos = [
            pos.max_angle_limit / 10.0
            for pos in limits.all_motor_angle_limit_max_spd.motor[1:7]
        ] + [10.0]

        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_observation(self) -> dict[str, Any]:
        js = self.piper.GetArmJointMsgs().joint_state
        g = self.piper.GetArmGripperMsgs()

        obs = {
            f"joint_{i}.pos": getattr(js, f"joint_{i}") / 1000.0 for i in range(1, 7)
        }
        obs["gripper.pos"] = (g.gripper_state.grippers_angle / 10000.0,)

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()

        return obs

    @check_if_not_connected
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Hardware handles control via master-slave CAN - just return the action
        return action

    @check_if_not_connected
    def disconnect(self) -> None:
        while self.piper.DisablePiper():
            time.sleep(0.01)

        self.piper.DisconnectPort()

        for cam in self.cameras.values():
            cam.disconnect()
