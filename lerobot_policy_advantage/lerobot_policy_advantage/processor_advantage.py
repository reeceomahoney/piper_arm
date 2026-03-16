"""Processor for the advantage-conditioned policy.

Delegates to SmolVLA's processor since the advantage policy wraps SmolVLA
and uses identical input/output preprocessing.
"""

from typing import Any

import torch
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from .configuration_advantage import AdvantageConfig


def make_advantage_pre_post_processors(
    config: AdvantageConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Construct pre/post processors by delegating to SmolVLA's processor."""
    smolvla_config = config.to_smolvla_config()
    return make_smolvla_pre_post_processors(smolvla_config, dataset_stats=dataset_stats)
