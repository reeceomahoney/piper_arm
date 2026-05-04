import time

from piper_sdk import C_PiperInterface_V2


def main():
    arms = {
        "left": C_PiperInterface_V2("can_arm_left"),
        "right": C_PiperInterface_V2("can_arm_right"),
    }

    for arm in arms.values():
        arm.ConnectPort()
    time.sleep(0.1)

    for arm in arms.values():
        while not arm.EnablePiper():
            time.sleep(0.01)

    for arm in arms.values():
        arm.ModeCtrl(0x01, 0x01, 30, 0x00)

    for _ in range(5):
        for arm in arms.values():
            arm.JointCtrl(0, 0, 0, 0, 0, 0)
        time.sleep(0.5)

    for arm in arms.values():
        arm.GripperCtrl(0, 1000, 0x01, 0)


if __name__ == "__main__":
    main()
