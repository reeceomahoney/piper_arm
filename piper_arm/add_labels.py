"""Add steps_remaining and success columns to a LeRobot dataset.

Iterates over the dataset's parquet files, computes steps_remaining
per episode (counting down from episode length - 1 to 0), and sets
success=True for every frame.

Usage:
    python -m piper_arm.add_labels
    python -m piper_arm.add_labels --dataset_repo_id lerobot/libero
"""

import json
from dataclasses import dataclass

import draccus
import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class AddLabelsConfig:
    dataset_repo_id: str = "lerobot/libero"
    dataset_root: str | None = None
    push_to_hub: bool = True
    output_repo_id: str | None = "reece-omahoney/libero"


@draccus.wrap()  # type: ignore[misc]
def main(cfg: AddLabelsConfig):
    ds_kwargs: dict = {"repo_id": cfg.dataset_repo_id}
    if cfg.dataset_root:
        ds_kwargs["root"] = cfg.dataset_root
    dataset = LeRobotDataset(**ds_kwargs)

    print(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    data_dir = dataset.root / "data"
    total_modified = 0

    for pq_path in sorted(data_dir.glob("*/*.parquet")):
        df = pd.read_parquet(pq_path)
        n = len(df)

        steps_remaining = np.empty(n, dtype=np.int32)
        success = np.ones(n, dtype=bool)

        for ep_idx in df["episode_index"].unique():
            mask = df["episode_index"] == ep_idx
            ep_len = mask.sum()
            steps_remaining[mask] = np.arange(ep_len - 1, -1, -1, dtype=np.int32)

        df["steps_remaining"] = steps_remaining
        df["success"] = success
        df.to_parquet(pq_path, compression="snappy", index=False)
        total_modified += n
        print(f"  {pq_path.name}: {n} rows")

    print(f"Total rows modified: {total_modified}")

    # Update info.json with new features
    info_path = dataset.root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    changed = False
    features = info.get("features", {})
    if "steps_remaining" not in features:
        features["steps_remaining"] = {
            "dtype": "int32",
            "shape": [1],
            "names": None,
        }
        changed = True
    if "success" not in features:
        features["success"] = {
            "dtype": "bool",
            "shape": [1],
            "names": None,
        }
        changed = True

    if changed:
        info["features"] = features
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        print("Updated info.json with steps_remaining and success features")

    if cfg.push_to_hub:
        repo_id = cfg.output_repo_id or cfg.dataset_repo_id
        dataset.repo_id = repo_id
        print(f"Pushing dataset to {repo_id}...")
        dataset.push_to_hub()

    print("Done.")


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
