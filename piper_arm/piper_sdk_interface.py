# Piper SDK interface for LeRobot integration

import time
import logging

log = logging.getLogger(__name__)

try:
    from piper_sdk import C_PiperInterface_V2
except Exception:
    C_PiperInterface_V2 = None
    log.debug(
        "piper_sdk not available at import time; use `pip install piper_sdk` if you need hardware access"
    )


class PiperSDKInterface:
    def __init__(self, port: str = "can0", enable_timeout: float = 5.0):
        if C_PiperInterface_V2 is None:
            raise ImportError(
                "piper_sdk is not installed. Install with `pip install piper_sdk`."
            )
        try:
            self.piper = C_PiperInterface_V2(port)
        except Exception as e:
            log.error(
                "Failed to initialize Piper SDK: %s. Did you activate the CAN interface?",
                e,
            )
            self.piper = None
            raise RuntimeError("Failed to initialize Piper SDK") from e

        try:
            self.piper.ConnectPort()
            time.sleep(0.1)  # wait for connection to establish
        except Exception as e:
            log.error("ConnectPort failed: %s", e)
            raise

        # reset the arm if it's not in idle state (safe resume)
        try:
            status = self.piper.GetArmStatus().arm_status
            log.debug(
                "Initial arm motion_status=%s ctrl_mode=%s",
                getattr(status, "motion_status", None),
                getattr(status, "ctrl_mode", None),
            )
            if status.motion_status != 0:
                self.piper.EmergencyStop(0x02)  # resume
            if status.ctrl_mode == 2:
                log.warning(
                    "Arm is in teaching mode (ctrl_mode==2). Attempting resume."
                )
                self.piper.EmergencyStop(0x02)
        except Exception as e:
            log.debug("Unable to read arm status: %s", e)

        # wait for EnablePiper with timeout to avoid infinite loop
        start = time.time()
        while True:
            try:
                ok = self.piper.EnablePiper()
            except Exception:
                ok = False
            if ok:
                break
            if time.time() - start > enable_timeout:
                raise TimeoutError(
                    f"EnablePiper timed out after {enable_timeout} seconds"
                )
            time.sleep(0.01)

        # Get the min and max positions for each joint and gripper
        try:
            angel_status = self.piper.GetAllMotorAngleLimitMaxSpd()
            # SDK motor list appears 1-indexed -> extract 1..6
            # Angle limits are in deci-degrees (0.1 deg). Convert to degrees for consistency.
            self.min_pos = [
                pos.min_angle_limit / 10.0
                for pos in angel_status.all_motor_angle_limit_max_spd.motor[1:7]
            ] + [0.0]
            self.max_pos = [
                pos.max_angle_limit / 10.0
                for pos in angel_status.all_motor_angle_limit_max_spd.motor[1:7]
            ] + [10.0]
        except Exception as e:
            log.warning("Could not read joint limits: %s", e)
            # sensible defaults to avoid crashes; keep lists length >=7
            self.min_pos = [-180.0] * 6 + [0.0]
            self.max_pos = [180.0] * 6 + [10.0]

    def get_status_deg(self) -> dict[str, float]:
        """Return joints in degrees and gripper in mm."""
        js = self.piper.GetArmJointMsgs().joint_state
        g = self.piper.GetArmGripperMsgs()
        out = {
            "joint_1.pos": js.joint_1 / 1000.0,
            "joint_2.pos": js.joint_2 / 1000.0,
            "joint_3.pos": js.joint_3 / 1000.0,
            "joint_4.pos": js.joint_4 / 1000.0,
            "joint_5.pos": js.joint_5 / 1000.0,
            "joint_6.pos": js.joint_6 / 1000.0,
        }
        # Convert gripper back from SDK unit to mm (SDK used *10000 when sending)
        try:
            out["gripper.pos"] = g.gripper_state.grippers_angle / 10000.0
        except Exception:
            pass
        return out

    def disconnect(self):
        try:
            # safe stop; values are SDK-specific, keep as-is but guarded
            self.piper.JointCtrl(0, 0, 0, 0, 25000, 0)
        except Exception:
            log.debug(
                "Disconnect: JointCtrl cleanup failed or piper already disconnected"
            )
