"""Advantage-conditioned policy configuration.

Extends SmolVLAConfig with advantage-specific parameters.
Registered as the "advantage" policy type for LeRobot plugin discovery.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("advantage")
@dataclass
class AdvantageConfig(SmolVLAConfig):
    fixed_advantage: bool = False
    advantage_dropout: float = 0.3
    guidance_scale: float = 1.0
    num_adv_tokens: int = 1
    use_text_advantage: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.fixed_advantage:
            self.advantage_dropout = 0.0
