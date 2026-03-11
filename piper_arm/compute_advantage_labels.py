"""Pre-compute binary advantage labels and add them to a LeRobot dataset.

Loads a trained value model, computes n-step TD advantages, then binarizes
per-sample advantages using per-task percentile thresholds. Adds an
`advantage_label` column to the dataset's parquet files.

N-step advantage: A(t) = sum_{k=0}^{N-1} r_{t+k} + V(t+N) - V(t)
where r = -1/max_ep_len per step. Falls back to MC return when the n-step
window extends past the episode boundary.

Usage:
    python -m piper_arm.compute_advantage_labels
"""

import json
from dataclasses import dataclass

import draccus
import numpy as np
import pandas as pd
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

from piper_arm.train_value import TrainValueConfig as TrainValueConfig
from piper_arm.train_value import load_value_preprocessor
from piper_arm.value_model import ValueConfig, ValueModel


@dataclass
class ComputeAdvantageLabelsConfig:
    value_checkpoint: str = "outputs/value/2026-03-10/14-29-58/checkpoint_40000.pt"
    pretrained_path: str = "reece-omahoney/smolvla-libero-16-chunk"
    dataset_repo_id: str = "reece-omahoney/libero-10-maha"
    dataset_root: str | None = None
    c_fail: float = 1000.0
    reward_type: str = "steps_remaining"  # "steps_remaining" or "maha_distance"
    n_step: int = 10
    advantage_percentile: float = 0.7
    batch_size: int = 64
    num_workers: int = 4
    push_to_hub: bool = True


def load_value_model(checkpoint_path: str, device: torch.device) -> ValueModel:
    """Load a trained value model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: ValueConfig = ckpt["config"].value
    model = ValueModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def compute_all_values(
    dataset: LeRobotDataset,
    value_model: ValueModel,
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
        if batch_idx % 100 == 0 or batch_idx == total_batches - 1:
            print(f"  [values] batch {batch_idx + 1}/{total_batches}", flush=True)

        batch = preprocessor(batch)
        with torch.no_grad():
            logits = value_model(batch)
            values = value_model.predict_value(logits)
        all_values.extend(values.cpu().tolist())

    return np.array(all_values, dtype=np.float64)


def compute_nstep_advantages(
    values: np.ndarray,
    steps_remaining: np.ndarray,
    success: np.ndarray,
    episode_index: np.ndarray,
    max_episode_length: int,
    n_step: int,
    c_fail: float,
) -> np.ndarray:
    """Compute n-step TD advantages for all samples.

    A(t) = sum_{k=0}^{N-1} r_{t+k} + V(t+N) - V(t)

    Per-step reward is -1/max_episode_length. When the n-step window extends
    past the episode end (steps_remaining < n_step), falls back to the MC
    return G(t) - V(t).

    Args:
        values: (N,) predicted values for each frame.
        steps_remaining: (N,) steps remaining per frame.
        success: (N,) bool success per frame.
        episode_index: (N,) episode index per frame.
        max_episode_length: max steps in any episode.
        n_step: number of lookahead steps N.
        c_fail: failure penalty for MC fallback.

    Returns:
        (N,) float array of advantages.
    """
    num_frames = len(values)
    advantages = np.zeros(num_frames, dtype=np.float64)
    step_reward = -1.0 / max_episode_length

    for i in range(num_frames):
        sr = int(steps_remaining[i])
        ep = int(episode_index[i])

        # Check that i + n_step lands in the same episode
        target = i + n_step
        in_episode = (
            sr >= n_step and target < num_frames and int(episode_index[target]) == ep
        )

        if in_episode:
            # N-step: A = sum of N step rewards + V(t+N) - V(t)
            n_step_return = n_step * step_reward + values[target]
            advantages[i] = n_step_return - values[i]
        else:
            # Near episode end: fall back to MC return
            succ = bool(success[i])
            if succ:
                mc_return = -sr / max_episode_length
            else:
                mc_return = max((-sr - c_fail + 1) / max_episode_length, -1.0)
            advantages[i] = mc_return - values[i]

    return advantages


def compute_nstep_advantages_maha(
    values: np.ndarray,
    maha_distance: np.ndarray,
    maha_norm: float,
    episode_index: np.ndarray,
    n_step: int,
) -> np.ndarray:
    """Compute n-step TD advantages using negative Mahalanobis distance rewards.

    Per-step reward: r_t = -maha_distance[t] / maha_norm

    N-step (when window stays in episode):
        A(t) = sum_{k=0}^{N-1} r_{t+k} + V(t+N) - V(t)

    MC fallback (near episode end):
        A(t) = sum_{k=t}^{T-1} r_k - V(t)

    No c_fail penalty — high maha distances naturally produce lower returns.

    Args:
        values: (N,) predicted values for each frame.
        maha_distance: (N,) Mahalanobis distance per frame.
        maha_norm: normalization constant (from value training checkpoint).
        episode_index: (N,) episode index per frame.
        n_step: number of lookahead steps N.

    Returns:
        (N,) float array of advantages.
    """
    num_frames = len(values)
    advantages = np.zeros(num_frames, dtype=np.float64)
    rewards = -maha_distance.astype(np.float64) / maha_norm

    for i in range(num_frames):
        ep = int(episode_index[i])
        target = i + n_step
        in_episode = target < num_frames and int(episode_index[target]) == ep

        if in_episode:
            n_step_reward_sum = rewards[i:target].sum()
            advantages[i] = n_step_reward_sum + values[target] - values[i]
        else:
            # MC fallback: sum remaining rewards in this episode
            j = i
            while j < num_frames and int(episode_index[j]) == ep:
                j += 1
            mc_return = rewards[i:j].sum()
            advantages[i] = mc_return - values[i]

    return advantages


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

    # Load dataset
    ds_kwargs: dict = {"repo_id": cfg.dataset_repo_id}
    if cfg.dataset_root:
        ds_kwargs["root"] = cfg.dataset_root
    dataset = LeRobotDataset(**ds_kwargs)

    all_steps = dataset.hf_dataset["steps_remaining"]
    max_episode_length = max(s.item() for s in all_steps) + 1
    print(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    print(f"Max episode length: {max_episode_length}")

    # Load value model & preprocessor
    print("Loading value model...")
    value_model: ValueModel = load_value_model(cfg.value_checkpoint, device)
    preprocessor = load_value_preprocessor(cfg.pretrained_path)

    # Compute V(s) for all samples
    print("Computing values for all samples...")
    values = compute_all_values(
        dataset,
        value_model,
        preprocessor,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # Extract episode metadata as arrays
    steps_remaining = np.array(
        [s.item() for s in dataset.hf_dataset["steps_remaining"]]
    )
    success = np.array([s.item() for s in dataset.hf_dataset["success"]])
    episode_index = np.array([s.item() for s in dataset.hf_dataset["episode_index"]])

    # Collect task strings (need a sequential pass through the dataset)
    print("Collecting task strings...")
    tasks: list[str] = []
    task_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    for batch in task_loader:
        tasks.extend(batch["task"])

    # Compute n-step advantages
    print(f"Computing {cfg.n_step}-step advantages...")
    if cfg.reward_type == "maha_distance":
        maha_distance = np.array(
            [s.item() for s in dataset.hf_dataset["maha_distance"]],
            dtype=np.float64,
        )
        ckpt = torch.load(cfg.value_checkpoint, map_location="cpu", weights_only=False)
        maha_norm = ckpt["maha_norm"]
        print(f"Loaded maha_norm={maha_norm:.4f} from checkpoint")
        advantages = compute_nstep_advantages_maha(
            values,
            maha_distance,
            maha_norm,
            episode_index,
            cfg.n_step,
        )
    else:
        advantages = compute_nstep_advantages(
            values,
            steps_remaining,
            success,
            episode_index,
            max_episode_length,
            cfg.n_step,
            cfg.c_fail,
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

    assert (
        len(labels) == dataset.num_frames
    ), f"Expected {dataset.num_frames} labels, got {len(labels)}"

    pct_positive = sum(labels) / len(labels) * 100
    print(
        f"Labels computed: {pct_positive:.1f}% positive ({sum(labels)}/{len(labels)})"
    )

    # Save by overwriting parquet files with the new column added
    data_dir = dataset.root / "data"
    print(f"Saving advantage_label to parquet files in {data_dir}...")
    offset = 0
    for pq_path in sorted(data_dir.glob("*/*.parquet")):
        df = pd.read_parquet(pq_path)
        n = len(df)
        df["advantage_label"] = labels[offset : offset + n]
        offset += n
        df.to_parquet(pq_path, compression="snappy", index=False)
    assert offset == len(
        labels
    ), f"Parquet files had {offset} rows, expected {len(labels)}"

    # Update info.json to register the new feature
    info_path = dataset.root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    if "advantage_label" not in info.get("features", {}):
        info["features"]["advantage_label"] = {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        print("Updated info.json with advantage_label feature")

    # Push to hub
    if cfg.push_to_hub:
        print("Pushing dataset...")
        dataset.push_to_hub()

    print("Done.")


if __name__ == "__main__":
    main()
