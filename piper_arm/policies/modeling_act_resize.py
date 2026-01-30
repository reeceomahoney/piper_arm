from lerobot.policies.act.modeling_act import ACTPolicy

from piper_arm.policies.configuration_act_resize import ACTResizeConfig


class ACTResizePolicy(ACTPolicy):
    config_class = ACTResizeConfig
    name = "act_resize"
