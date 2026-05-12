"""Push a trained checkpoint to HuggingFace Hub."""

import argparse
import logging
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.import_utils import register_third_party_plugins

from distal.value_model import RECAPValueNetwork

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def push_value(checkpoint_dir: Path, repo_id: str, private: bool):
    from huggingface_hub import upload_file
    from lerobot.processor.pipeline import PolicyProcessorPipeline

    model = RECAPValueNetwork.from_pretrained(str(checkpoint_dir))
    logging.info("Loaded value model from %s", checkpoint_dir)
    model.push_to_hub(repo_id, private=private)

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        str(checkpoint_dir), config_filename="policy_preprocessor.json"
    )
    preprocessor.push_to_hub(repo_id)

    train_config_path = checkpoint_dir / "train_config.json"
    if train_config_path.exists():
        upload_file(
            path_or_fileobj=str(train_config_path),
            path_in_repo="train_config.json",
            repo_id=repo_id,
        )
        logging.info("Uploaded train_config.json")


def push_policy(checkpoint_dir: Path, repo_id: str, private: bool):
    from huggingface_hub import HfApi

    policy_config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    logging.info("Loaded %s policy from %s", policy_config.type, checkpoint_dir)

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(checkpoint_dir),
        commit_message="Upload policy weights, train config and processors",
        allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
        ignore_patterns=["*.tmp", "*.log"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Push a trained checkpoint to HuggingFace Hub"
    )
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("repo_id", type=str)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    register_third_party_plugins()
    checkpoint_dir = args.checkpoint_dir.resolve()
    policy_config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    logging.info("Detected policy type: %s", policy_config.type)

    if policy_config.type == "recap_value":
        push_value(checkpoint_dir, args.repo_id, args.private)
    else:
        push_policy(checkpoint_dir, args.repo_id, args.private)

    logging.info("Pushed to https://huggingface.co/%s", args.repo_id)


if __name__ == "__main__":
    main()
