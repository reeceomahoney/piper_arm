"""Pre-compute binary advantage labels and add them to a LeRobot dataset.

Loads a trained value model, computes n-step TD advantages, then binarizes
per-sample advantages using per-task percentile thresholds. Adds an
`observation.language.advantage_label` column with integer labels (0 or 1)
to the dataset's parquet files.

N-step advantage uses shared reward helpers and falls back to normalized MC
returns when the n-step window extends past the episode boundary.
"""

import json
from collections import Counter
from dataclasses import dataclass

import draccus
import numpy as np
import pandas as pd
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.pipeline import PolicyProcessorPipeline
from torch.utils.data import DataLoader

import lerobot_policy_advantage as lerobot_policy_advantage
from distal.rewards import compute_nstep_advantages
from distal.train_value import _load_episode_success_from_dataset
from distal.value_model import RECAPValueNetwork


@dataclass
class ComputeAdvantageLabelsConfig:
    value_checkpoint: str = "reece-omahoney/value-steps-paligemma"
    dataset_repo_id: str = "reece-omahoney/libero-10"
    new_dataset_repo_id: str = "reece-omahoney/libero-10-adv-steps-paligemma"
    push_to_hub: bool = True
    device: str = "cuda"
    n_step: int = 50
    advantage_percentile: float = 0.7
    batch_size: int = 256
    num_workers: int = 16
    c_fail: float = 500.0
    gamma: float = 1.0


def compute_rewards_and_returns(
    dataset: LeRobotDataset,
    success_by_episode: dict[int, int],
    c_fail: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-frame normalized rewards and returns matching value training.

    Uses the same reward scheme as train_value: -1 per step, 0 or -c_fail at
    terminal, normalized per-task by max episode length, clipped to [-1, 0].
    """
    num_frames = dataset.num_frames
    rewards = np.zeros(num_frames, dtype=np.float64)
    returns = np.zeros(num_frames, dtype=np.float64)

    episode_info: list[tuple[int, int, int, str]] = []
    for ep_idx in range(dataset.meta.total_episodes):
        ep_data = dataset.meta.episodes[ep_idx]
        start = int(ep_data["dataset_from_index"])
        end = int(ep_data["dataset_to_index"])
        length = end - start

        if "task_index" in ep_data and dataset.meta.tasks is not None:
            task = str(dataset.meta.tasks.iloc[int(ep_data["task_index"])].name)
        elif "tasks" in ep_data and ep_data["tasks"]:
            tasks = ep_data["tasks"]
            task = tasks if isinstance(tasks, str) else str(tasks[0])
        else:
            raise ValueError(f"Episode {ep_idx} has no task information")

        episode_info.append((start, end, length, task))

    task_max_len: dict[str, int] = {}
    for _, _, length, task in episode_info:
        task_max_len[task] = max(task_max_len.get(task, 1), length)

    for ep_idx, (start, end, length, task) in enumerate(episode_info):
        success = bool(success_by_episode[ep_idx])
        ep_rewards = np.full(length, -1.0, dtype=np.float64)
        ep_rewards[-1] = 0.0 if success else -float(c_fail)

        ep_returns = np.flip(np.cumsum(np.flip(ep_rewards)))

        norm = float(task_max_len[task])
        rewards[start:end] = ep_rewards / norm
        returns[start:end] = np.clip(ep_returns / norm, -1.0, 0.0)

    return rewards, returns


def compute_all_values(
    dataset: LeRobotDataset,
    value_model: RECAPValueNetwork,
    preprocessor,
    batch_size: int = 64,
    num_workers: int = 4,
) -> np.ndarray:
    """Compute V(s) for every sample in the dataset.

    Returns:
        (N,) float array of predicted values, one per frame.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    all_values: list[float] = []
    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader):
        if batch_idx % 20 == 0 or batch_idx == total_batches - 1:
            print(f"  [values] batch {batch_idx + 1}/{total_batches}", flush=True)

        batch = preprocessor(batch)
        with torch.no_grad():
            values = value_model.predict_value(batch)
        all_values.extend(values.cpu().tolist())

    return np.array(all_values, dtype=np.float64)


def compute_thresholds_from_advantages(
    advantages: np.ndarray,
    tasks: list[str],
    percentile: float,
) -> dict[str, float]:
    """Compute per-task advantage thresholds at the given percentile."""
    task_advantages: dict[str, list[float]] = {}
    for i, task in enumerate(tasks):
        if task not in task_advantages:
            task_advantages[task] = []
        task_advantages[task].append(advantages[i])

    thresholds: dict[str, float] = {}
    for task, advs in task_advantages.items():
        advs_t = torch.tensor(advs)
        thresholds[task] = torch.quantile(advs_t, percentile).item()
        print(
            f"  Task: {task[:60]:60s}  threshold={thresholds[task]:.4f}  n={len(advs)}"
        )
    return thresholds


def binarize_advantages(
    advantages: np.ndarray,
    tasks: list[str],
    thresholds: dict[str, float],
) -> list[int]:
    """Binarize advantages using per-task thresholds."""
    labels: list[int] = []
    for i, task in enumerate(tasks):
        threshold = thresholds.get(task, 0.0)
        labels.append(1 if advantages[i] > threshold else 0)
    return labels


@draccus.wrap()
def main(cfg: ComputeAdvantageLabelsConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LeRobotDataset(cfg.dataset_repo_id)

    episode_index_all = np.array(
        [s.item() for s in dataset.hf_dataset["episode_index"]]
    )
    ep_lengths = Counter(episode_index_all.tolist())
    max_episode_length = max(ep_lengths.values())
    print(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    print(f"Max episode length: {max_episode_length}")

    # Load value model & preprocessor
    print("Loading value model...")
    value_model = RECAPValueNetwork.from_pretrained(cfg.value_checkpoint)
    value_model.to(device).eval()
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        cfg.value_checkpoint, config_filename="policy_preprocessor.json"
    )

    # Compute V(s) for all samples
    print("Computing values for all samples...")
    values = compute_all_values(
        dataset,
        value_model,
        preprocessor,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # Get per-frame task strings
    task_index = np.array([s.item() for s in dataset.hf_dataset["task_index"]])
    task_map = {v: k for k, v in dataset.meta.tasks["task_index"].items()}
    tasks = [task_map[i] for i in task_index]

    # Compute rewards and returns from per-episode success labels
    success_by_episode = _load_episode_success_from_dataset(dataset)
    print(f"Loaded success labels for {len(success_by_episode)} episodes")

    rewards, returns = compute_rewards_and_returns(
        dataset, success_by_episode, cfg.c_fail
    )
    print(f"Computed rewards/returns (c_fail={cfg.c_fail}, gamma={cfg.gamma})")

    # Compute n-step advantages
    print(f"Computing {cfg.n_step}-step advantages...")
    advantages = compute_nstep_advantages(
        values,
        rewards,
        returns,
        episode_index_all,
        cfg.n_step,
        cfg.gamma,
    )

    # Compute per-task thresholds and binarize
    print("Computing per-task advantage thresholds...")
    thresholds = compute_thresholds_from_advantages(
        advantages,
        tasks,
        cfg.advantage_percentile,
    )

    print("Binarizing advantage labels...")
    labels = binarize_advantages(advantages, tasks, thresholds)

    assert len(labels) == dataset.num_frames, (
        f"Expected {dataset.num_frames} labels, got {len(labels)}"
    )

    num_positive = sum(labels)
    pct_positive = num_positive / len(labels) * 100
    print(
        f"Labels computed: {pct_positive:.1f}% positive ({num_positive}/{len(labels)})"
    )

    # Save by overwriting parquet files with the new column added
    col_name = "observation.language.advantage_label"
    data_dir = dataset.root / "data"
    print(f"Saving {col_name} to parquet files in {data_dir}...")
    offset = 0
    for pq_path in sorted(data_dir.glob("*/*.parquet")):
        df = pd.read_parquet(pq_path)
        n = len(df)
        df[col_name] = labels[offset : offset + n]
        offset += n
        df.to_parquet(pq_path, compression="snappy", index=False)
    assert offset == len(labels), (
        f"Parquet files had {offset} rows, expected {len(labels)}"
    )

    # Update info.json to register the new feature
    info_path = dataset.root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    if col_name not in info.get("features", {}):
        info["features"][col_name] = {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        print(f"Updated info.json with {col_name} feature")

    # Push to hub
    if cfg.push_to_hub:
        print("Pushing dataset...")
        dataset.repo_id = cfg.new_dataset_repo_id
        dataset.push_to_hub()

    print("Done.")


if __name__ == "__main__":
    main()
