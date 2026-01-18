from piper_sdk import C_PiperInterface_V2


def main():
    piper = C_PiperInterface_V2()
    piper.ConnectPort()
    # piper.MasterSlaveConfig(0xFA, 0, 0, 0)


if __name__ == "__main__":
    main()
