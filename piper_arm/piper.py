from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot

from .config_piper import PiperConfig
from .piper_sdk_interface import PiperSDKInterface


class Piper(Robot):
    config_class = PiperConfig
    name = "piper"

    def __init__(self, config: PiperConfig):
        super().__init__(config)
        self.config = config
        # Lazily initialize the SDK interface in connect()
        self._iface: PiperSDKInterface | None = None
        self.cameras = make_cameras_from_configs(config.cameras) if config.cameras else {}

    @property
    def is_connected(self) -> bool:
        return (
            self._iface is not None
            and getattr(self._iface, "piper", None) is not None
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{j}.pos": float for j in self.config.joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {k: (c.height, c.width, 3) for k, c in self.cameras.items()}

    @property
    def observation_features(self) -> dict:
        ft = {**self._motors_ft, **self._cameras_ft}
        if self.config.include_gripper:
            ft["gripper.pos"] = float
        return ft

    @property
    def action_features(self) -> dict:
        ft = {f"{name}.pos": float for name in self.config.joint_names}
        if self.config.include_gripper:
            ft["gripper.pos"] = float
        return ft

    def connect(self, calibrate: bool = True) -> None:
        # Initialize SDK interface on demand
        if self._iface is None:
            self._iface = PiperSDKInterface(
                port=self.config.can_interface,
                enable_timeout=self.config.enable_timeout,
            )
        for cam in self.cameras.values():
            cam.connect()
        self.configure()

    def disconnect(self) -> None:
        if self._iface is not None:
            # Optionally perform a safe resume/stop here
            self._iface = None
        for cam in self.cameras.values():
            cam.disconnect()

    @property
    def is_calibrated(self) -> bool:  # type: ignore[override]
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _get_hw_limits(self) -> tuple[list[float], list[float]]:
        if self._iface is None:
            raise RuntimeError("Piper SDK interface not available")
        min_pos = getattr(self._iface, "min_pos", None)
        max_pos = getattr(self._iface, "max_pos", None)
        if not isinstance(min_pos, list) or not isinstance(max_pos, list) or len(min_pos) < 7 or len(max_pos) < 7:
            raise RuntimeError("Piper SDK limits unavailable")
        return min_pos, max_pos

    def _get_oriented_limits(self) -> tuple[list[float], list[float]]:
        min_pos, max_pos = self._get_hw_limits()
        oriented_min: list[float] = []
        oriented_max: list[float] = []
        for idx, sign in enumerate(self.config.joint_signs):
            hw_min = min_pos[idx]
            hw_max = max_pos[idx]
            if sign >= 0:
                oriented_min.append(hw_min)
                oriented_max.append(hw_max)
            else:
                oriented_min.append(-hw_max)
                oriented_max.append(-hw_min)
        return oriented_min, oriented_max

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected or self._iface is None:
            raise ConnectionError(f"{self} is not connected.")
        status = self._iface.get_status_deg()

        if not self.config.use_degrees:
            oriented_min, oriented_max = self._get_oriented_limits()

            def deg_to_pct(deg: float, idx: int) -> float:
                rng_min = oriented_min[idx]
                rng_max = oriented_max[idx]
                if rng_max <= rng_min:
                    return 0.0
                pct = (deg - rng_min) / (rng_max - rng_min) * 200.0 - 100.0
                return max(-100.0, min(100.0, pct))
        else:
            def deg_to_pct(deg: float, idx: int) -> float:  # type: ignore[no-redef]
                return deg

        obs = {}
        for i, name in enumerate(self.config.joint_names, start=1):
            deg = status[f"joint_{i}.pos"] * self.config.joint_signs[i - 1]
            obs[f"{name}.pos"] = deg if self.config.use_degrees else deg_to_pct(deg, i - 1)

        if self.config.include_gripper and "gripper.pos" in status:
            grip_mm = status["gripper.pos"]
            if self.config.use_degrees:
                obs["gripper.pos"] = grip_mm
            else:
                g_min = self._iface.min_pos[6]
                g_max = self._iface.max_pos[6]
                if g_max > g_min:
                    obs["gripper.pos"] = (grip_mm - g_min) / (g_max - g_min) * 100.0
                else:
                    obs["gripper.pos"] = 0.0

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Hardware handles control via master-slave CAN - just return the action
        return action