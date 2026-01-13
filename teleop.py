import time
from piper_sdk.interface.piper_interface_v2 import C_PiperInterface_V2


def main():
    piper_1 = C_PiperInterface_V2("can_arm1")
    piper_1.ConnectPort()
    piper_1.MasterSlaveConfig(0xFA, 0, 0, 0)

    piper_2 = C_PiperInterface_V2("can_arm2")
    piper_2.ConnectPort()
    piper_2.MasterSlaveConfig(0xFC, 0, 0, 0)
    piper_2.MotionCtrl_1(0x02, 0, 0)

    while not piper_2.EnablePiper():
        time.sleep(0.01)

    # Low pass filter
    smooth_joint_list = [0] * 6
    alpha = 0.1

    print("Starting Teleop")
    try:
        while True:
            target_joints = piper_1.GetArmJointCtrl()
            target_gripper = piper_1.GetArmGripperCtrl().gripper_ctrl.grippers_angle
            joint_list = [
                getattr(target_joints.joint_ctrl, f"joint_{i}") for i in range(1, 7)
            ]
            # Smooth joint movements
            for i in range(6):
                smooth_joint_list[i] = int(
                    smooth_joint_list[i] * (1 - alpha) + joint_list[i] * alpha
                )

            piper_2.MotionCtrl_2(0x01, 0x01, 100, 0xAD)
            piper_2.JointCtrl(*smooth_joint_list)
            piper_2.GripperCtrl(abs(target_gripper), 1000, 0x01, 0)
    finally:
        print("Disabling Arm")
        piper_2.DisableArm()


if __name__ == "__main__":
    main()
