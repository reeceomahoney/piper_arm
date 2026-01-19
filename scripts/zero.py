import time
from piper_sdk import C_PiperInterface_V2


def main():
    piper = C_PiperInterface_V2()
    piper.ConnectPort()
    # piper.MasterSlaveConfig(0xFA, 0, 0, 0)

    while not piper.EnablePiper():
        time.sleep(0.01)

    piper.ModeCtrl(0x01, 0x01, 30, 0x00)
    piper.JointCtrl(0, 0, 0, 0, 0, 0)
    piper.GripperCtrl(0, 1000, 0x01, 0)


if __name__ == "__main__":
    main()
