#!/usr/bin/env python
"""
Push a trained lerobot checkpoint to HuggingFace Hub.

Usage:
    python -m piper_arm.scripts.push_to_hub \
        --checkpoint-dir outputs/train/my_policy/checkpoints/last \
        --repo-id my-username/my-policy
"""

import argparse
import logging
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    parser = argparse.ArgumentParser(
        description="Push a trained lerobot checkpoint to HuggingFace Hub"
    )
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("repo_id", type=str)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.resolve()

    # Load configs
    policy_config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    train_config = TrainPipelineConfig.from_pretrained(checkpoint_dir)

    # Override repo_id and private settings
    policy_config.repo_id = args.repo_id
    policy_config.private = args.private

    # Load policy
    policy_cls = get_policy_class(policy_config.type)
    policy = policy_cls.from_pretrained(
        pretrained_name_or_path=str(checkpoint_dir), config=policy_config
    )
    logging.info(f"Loaded {policy_config.type} policy")

    # Load processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_config, pretrained_path=str(checkpoint_dir)
    )
    logging.info("Loaded processors")

    # Push to hub (same as lerobot training script)
    policy.push_model_to_hub(train_config)
    preprocessor.push_to_hub(args.repo_id)
    postprocessor.push_to_hub(args.repo_id)

    logging.info(f"Pushed to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
