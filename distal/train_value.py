#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train/val script for the standalone RECAP distributional value network.

Reads per-episode success labels from the dataset's ``success`` column
(populated by ``distal/collect.py``).
"""

import logging
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import _default_decoder_cache
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_json
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import cycle, init_logging
from torch.utils.data import DataLoader, Dataset

from distal.rewards.configs import KnnRewardConfig, RewardConfig
from distal.value_model import (
    RECAPValueConfig,
    RECAPValueNetwork,
    build_value_preprocessor,
)


@dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    task: str
    start_index: int
    end_index: int
    length: int


@dataclass(frozen=True)
class FrameTarget:
    frame_index: int
    episode_index: int
    success: int
    task: str
    target_value: float
    target_bin: int


@dataclass(frozen=True)
class ValidationFramePrediction:
    frame_index: int
    success: int
    target_value: float
    target_bin: int
    predicted_bin: int
    reconstructed_value: float
    predicted_probs: np.ndarray


@dataclass
class RECAPValueTrainingConfig:
    """Configuration for RECAP value-network train/val."""

    job_name: str = "value-knn-libero"
    repo_id: str = "reece-omahoney/pi05-libero-10"
    train_steps: int = 20_000
    batch_size: int = 64
    num_workers: int = 8
    learning_rate: float = 2.5e-5
    val_split_ratio: float = 0.1
    seed: int = 42
    device: str = "cuda"
    log_every_n_steps: int = 100
    plot_every_n_train_steps: int = 200
    save_every_n_steps: int = 1000
    max_val_steps: int | None = 20

    # Early stopping is counted in validation events, which coincide with
    # log steps. With log_every_n_steps=100 and patience=20, that is ~2000
    # train steps without improvement.
    early_stopping_patience: int | None = 20
    early_stopping_min_delta: float = 0.001

    # Value target construction
    c_fail: float = 50.0

    # Per-step reward source. See RewardConfig subclasses (steps / maha / knn).
    reward: RewardConfig = field(default_factory=KnnRewardConfig)

    # Value network architecture (overrides RECAPValueConfig defaults).
    model: RECAPValueConfig = field(
        default_factory=lambda: RECAPValueConfig(
            gradient_checkpointing=False, compile_model=True
        )
    )

    # Hub push for trained value network
    hub_user: str = "reece-omahoney"
    push_to_hub: bool = True

    # Weights & Biases (optional; set wandb_project to enable)
    wandb_project: str | None = "distal-value"
    wandb_entity: str | None = None

    @property
    def value_repo_id(self) -> str:
        return f"{self.hub_user}/{self.job_name}"


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    mins, secs = divmod(total_seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:d}h{mins:02d}m{secs:02d}s"
    if mins > 0:
        return f"{mins:d}m{secs:02d}s"
    return f"{secs:d}s"


def is_known_video_validation_error(error: Exception) -> bool:
    message = str(error)
    return (
        "Could not push packet to decoder" in message
        or "Invalid data found when processing input" in message
        or "FrameTimestampError" in message
        or "tolerance_s=" in message
        or "Failed to decode frame" in message
        or "Invalid frame index=" in message
        or "no more frames left to decode" in message
    )


def load_episode_success_from_dataset(dataset: LeRobotDataset) -> dict[int, int]:
    """Read the per-episode ``success`` label written by ``collect.py``."""
    success_map: dict[int, int] = {}
    for ep_idx, success in zip(
        dataset.hf_dataset["episode_index"], dataset.hf_dataset["success"]
    ):
        ep_idx = int(ep_idx)
        if ep_idx not in success_map:
            success_map[ep_idx] = int(bool(success))
    return success_map


def selected_episode_indices(dataset: LeRobotDataset) -> list[int]:
    if dataset.episodes is None:
        return list(range(dataset.meta.total_episodes))
    return [int(ep_idx) for ep_idx in dataset.episodes]


def build_episode_infos(dataset: LeRobotDataset) -> dict[int, EpisodeInfo]:
    episode_infos: dict[int, EpisodeInfo] = {}
    for ep_idx in selected_episode_indices(dataset):
        ep_data = dataset.meta.episodes[ep_idx]
        start_index = int(ep_data["dataset_from_index"])
        end_index = int(ep_data["dataset_to_index"])
        length = max(1, end_index - start_index)

        task: str | None = None
        if "task_index" in ep_data and dataset.meta.tasks is not None:
            task_index = int(ep_data["task_index"])
            task = str(dataset.meta.tasks.iloc[task_index].name)
        elif "tasks" in ep_data:
            tasks = ep_data["tasks"]
            if isinstance(tasks, str):
                task = tasks
            elif tasks:
                task = str(tasks[0])

        if task is None:
            available_fields = sorted(ep_data.keys())
            raise ValueError(
                f"Episode {ep_idx} metadata is missing task information. "
                f"Expected either 'task_index' or non-empty 'tasks'. "
                f"Available fields: {available_fields}"
            )

        episode_infos[ep_idx] = EpisodeInfo(
            episode_index=ep_idx,
            task=task,
            start_index=start_index,
            end_index=end_index,
            length=length,
        )
    return episode_infos


def discretize_values(
    normalized_returns: torch.Tensor, num_value_bins: int
) -> torch.Tensor:
    bin_edges = torch.linspace(
        -1.0,
        0.0,
        num_value_bins + 1,
        dtype=normalized_returns.dtype,
        device=normalized_returns.device,
    )
    bin_ids = torch.bucketize(normalized_returns, bin_edges[1:], right=False)
    return bin_ids.clamp(min=0, max=num_value_bins - 1).to(torch.long)


def build_frame_targets(
    dataset: LeRobotDataset,
    success_by_episode: dict[int, int],
    c_fail: float,
    num_value_bins: int,
    step_rewards: dict[int, float] | None = None,
) -> list[FrameTarget]:
    episode_infos = build_episode_infos(dataset)

    task_max_episode_len: dict[str, int] = {}
    for info in episode_infos.values():
        task_max_episode_len[info.task] = max(
            task_max_episode_len.get(info.task, 1), info.length
        )

    abs_to_rel_idx = {
        int(abs_idx): rel_idx
        for rel_idx, abs_idx in enumerate(dataset.hf_dataset["index"])
    }

    missing_episode_labels = sorted(set(episode_infos) - set(success_by_episode))
    if missing_episode_labels:
        raise ValueError(
            "Dataset is missing success labels for episode indices: "
            f"{missing_episode_labels[:20]}"
        )

    frame_targets: list[FrameTarget] = []
    for ep_idx, info in episode_infos.items():
        success = bool(success_by_episode[ep_idx])
        if step_rewards is None:
            rewards = torch.full((info.length,), -1.0, dtype=torch.float32)
        else:
            rewards = torch.tensor(
                [
                    step_rewards[abs_idx]
                    for abs_idx in range(info.start_index, info.end_index)
                ],
                dtype=torch.float32,
            )
        rewards[-1] = 0.0 if success else -float(c_fail)

        returns = torch.flip(
            torch.cumsum(torch.flip(rewards, dims=[0]), dim=0), dims=[0]
        )
        max_len_for_task = float(task_max_episode_len[info.task])
        normalized_returns = torch.clamp(returns / max_len_for_task, min=-1.0, max=0.0)
        target_bins = discretize_values(
            normalized_returns, num_value_bins=num_value_bins
        )

        for offset, abs_index in enumerate(range(info.start_index, info.end_index)):
            if abs_index not in abs_to_rel_idx:
                continue
            rel_idx = abs_to_rel_idx[abs_index]
            frame_targets.append(
                FrameTarget(
                    frame_index=rel_idx,
                    episode_index=ep_idx,
                    success=int(success),
                    task=info.task,
                    target_value=float(normalized_returns[offset]),
                    target_bin=int(target_bins[offset]),
                )
            )

    if not frame_targets:
        raise ValueError(
            "No frame targets were constructed. Check dataset episodes and CSV labels."
        )

    return frame_targets


def split_train_val_targets(
    frame_targets: list[FrameTarget],
    val_ratio: float,
    seed: int,
) -> tuple[list[FrameTarget], list[FrameTarget]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_split_ratio must be in [0, 1). Got {val_ratio}.")

    episode_success: dict[int, int] = {
        target.episode_index: target.success for target in frame_targets
    }
    if len(episode_success) < 2:
        raise ValueError(
            "At least 2 labeled episodes are required for a train/val split."
        )

    success_ids = sorted(ep for ep, s in episode_success.items() if s == 1)
    failure_ids = sorted(ep for ep, s in episode_success.items() if s == 0)

    rng = random.Random(seed)
    rng.shuffle(success_ids)
    rng.shuffle(failure_ids)

    val_episode_ids: set[int] = set()
    for ids in (success_ids, failure_ids):
        if not ids:
            continue
        n_val = max(1, round(len(ids) * val_ratio))
        n_val = min(n_val, len(ids) - 1)  # leave at least 1 for train
        val_episode_ids.update(ids[:n_val])

    train_episode_ids = set(episode_success) - val_episode_ids
    if not val_episode_ids or not train_episode_ids:
        raise ValueError(
            "Train/val split produced an empty partition. Adjust val_split_ratio."
        )

    train_targets = [t for t in frame_targets if t.episode_index in train_episode_ids]
    val_targets = [t for t in frame_targets if t.episode_index in val_episode_ids]
    return train_targets, val_targets


class RECAPFrameSupervisionDataset(Dataset):
    """Frame-level dataset with paper-faithful return-bin supervision.

    Returns raw LeRobot frames (with per-camera image keys) augmented with
    value-training metadata.  Preprocessing (normalisation, state
    discretisation, tokenisation) is handled externally by the pi05
    preprocessor pipeline.
    """

    MAX_DECODE_RETRIES = 5
    RETRY_BASE_DELAY_S = 0.1

    def __init__(
        self,
        base_dataset: LeRobotDataset,
        frame_targets: list[FrameTarget],
    ):
        self.base_dataset = base_dataset
        self.frame_targets = frame_targets

    def __len__(self) -> int:
        return len(self.frame_targets)

    def decode_frame(self, frame_index: int) -> dict | None:
        """Try to decode a single frame, returning None on persistent failure."""
        for attempt in range(self.MAX_DECODE_RETRIES):
            try:
                return self.base_dataset[frame_index]
            except IndexError as exc:
                if not is_known_video_validation_error(exc):
                    raise
                return None
            except RuntimeError as exc:
                if not is_known_video_validation_error(exc):
                    raise
                time.sleep(self.RETRY_BASE_DELAY_S * (2**attempt))
        return None

    def __getitem__(self, index: int) -> dict:
        target = self.frame_targets[index]

        frame = self.decode_frame(target.frame_index)
        if frame is None:
            logging.warning(
                f"Permanently failed to decode frame {target.frame_index} "
                f"(episode {target.episode_index}); substituting a random frame."
            )
            for _ in range(self.MAX_DECODE_RETRIES):
                alt_index = random.randint(0, len(self.frame_targets) - 1)
                alt_target = self.frame_targets[alt_index]
                frame = self.decode_frame(alt_target.frame_index)
                if frame is not None:
                    target = alt_target
                    break
            else:
                raise RuntimeError(
                    f"Failed to decode frame {target.frame_index} and "
                    f"{self.MAX_DECODE_RETRIES} random substitutes"
                )

        frame["target_bin"] = target.target_bin
        frame["target_value"] = target.target_value
        frame["success"] = target.success
        frame["frame_index"] = target.frame_index
        return frame


TRAINING_METADATA_KEYS = ("target_bin", "target_value", "success", "frame_index")

VAL_PLOT_NUM_EPISODES = 4
VAL_PLOT_NUM_FRAMES = 8


def preprocess_batch(batch: dict, preprocessor) -> dict:
    """Apply the preprocessor while preserving training metadata keys."""
    preserved = {k: batch.pop(k) for k in TRAINING_METADATA_KEYS if k in batch}
    batch = preprocessor(batch)
    device = next(v.device for v in batch.values() if isinstance(v, torch.Tensor))
    for k, v in preserved.items():
        batch[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return batch


def select_validation_plot_episode_ids(
    frame_targets: list[FrameTarget], max_episodes: int
) -> list[int]:
    if max_episodes <= 0:
        return []

    episode_success = {t.episode_index: t.success for t in frame_targets}
    success_ids = sorted(ep for ep, s in episode_success.items() if s == 1)
    failure_ids = sorted(ep for ep, s in episode_success.items() if s == 0)

    selected: list[int] = []
    if success_ids:
        selected.append(success_ids[0])
    if failure_ids and len(selected) < max_episodes:
        selected.append(failure_ids[0])

    for ep_id in sorted(episode_success):
        if len(selected) >= max_episodes:
            break
        if ep_id not in selected:
            selected.append(ep_id)
    return selected


def sample_preview_frame_indices(
    frame_indices: list[int], num_frames: int
) -> list[int]:
    if not frame_indices or num_frames <= 0:
        return []
    if len(frame_indices) <= num_frames:
        return frame_indices

    sample_positions = np.linspace(0, len(frame_indices) - 1, num_frames, dtype=int)
    return [frame_indices[pos] for pos in sample_positions.tolist()]


def prepare_plot_frame(
    frame: dict, camera_keys: list[str], image_size: int
) -> np.ndarray:
    image: torch.Tensor | None = None
    for key in camera_keys:
        if key in frame:
            image = frame[key].float()
            break
    if image is None:
        image = torch.zeros(3, image_size, image_size)
    image = F.interpolate(
        image.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return np.clip(image.permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)


def save_validation_episode_plot(
    dataset: LeRobotDataset,
    episode_index: int,
    predictions: list[ValidationFramePrediction],
    output_path: Path,
    num_preview_frames: int,
    num_value_bins: int,
    image_size: int,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib is unavailable; skipping validation plots.")
        return False

    if not predictions:
        return False

    predictions = sorted(predictions, key=lambda x: x.frame_index)
    success = predictions[0].success
    status = "SUCCESS" if success == 1 else "FAIL"

    frame_indices = [pred.frame_index for pred in predictions]
    target_values = np.array(
        [pred.target_value for pred in predictions], dtype=np.float32
    )
    reconstructed_values = np.array(
        [pred.reconstructed_value for pred in predictions], dtype=np.float32
    )

    preview_indices = sample_preview_frame_indices(
        frame_indices=frame_indices, num_frames=num_preview_frames
    )
    preview_images: list[np.ndarray] = []
    preview_steps: list[int] = []
    camera_keys = list(dataset.meta.camera_keys)
    frame_to_step = {frame_idx: idx for idx, frame_idx in enumerate(frame_indices)}
    for frame_idx in preview_indices:
        frame = dataset[frame_idx]
        preview_images.append(
            prepare_plot_frame(
                frame=frame, camera_keys=camera_keys, image_size=image_size
            )
        )
        preview_steps.append(frame_to_step[frame_idx])

    time_steps = np.arange(len(predictions), dtype=np.int64)

    fig = plt.figure(figsize=(15, 6), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.6])

    top_grid = grid[0].subgridspec(1, max(1, len(preview_images)))
    if preview_images:
        for idx, image in enumerate(preview_images):
            ax = fig.add_subplot(top_grid[0, idx])
            ax.imshow(image)
            ax.set_title(f"t={preview_steps[idx]}", fontsize=9)
            ax.axis("off")
    else:
        ax = fig.add_subplot(top_grid[0, 0])
        ax.text(0.5, 0.5, "No preview frames", ha="center", va="center")
        ax.axis("off")

    ax_return = fig.add_subplot(grid[1])
    ax_return.plot(
        time_steps,
        target_values,
        color="red",
        linewidth=2,
        label="Labeled expected return",
    )
    ax_return.plot(
        time_steps,
        reconstructed_values,
        color="green",
        linewidth=2,
        label="Reconstructed return E[v|p(bin)]",
    )
    ax_return.set_ylim(-1.05, 0.05)
    ax_return.set_ylabel("Normalized return")
    ax_return.set_xlabel("Trajectory step")
    ax_return.grid(alpha=0.3)
    ax_return.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        f"Validation Episode {episode_index} ({status})", fontsize=14, fontweight="bold"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return True


def train_step(
    model: RECAPValueNetwork,
    batch: dict,
    preprocessor: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    max_grad_norm: float,
) -> dict[str, float]:
    batch = preprocess_batch(batch, preprocessor)
    # Strip non-tensor entries (e.g. raw "task" strings) so torch.compile's
    # dynamo doesn't specialize on their values and recompile per task.
    model_batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}

    loss, outputs = model(model_batch)
    loss.backward()
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if scheduler is not None:
        scheduler.step()

    with torch.no_grad():
        pred_bins = outputs["value_logits"].argmax(dim=-1)
        expected_value = outputs["expected_value"].squeeze(-1)
        acc = (pred_bins == batch["target_bin"]).float().mean()
        mae = torch.abs(expected_value - batch["target_value"]).mean()

    return {
        "loss": float(loss.item()),
        "bin_acc": float(acc.item()),
        "value_mae": float(mae.item()),
    }


def validate(
    model: RECAPValueNetwork,
    loader: DataLoader,
    preprocessor: Any,
    max_steps: int | None = None,
    collect_episode_ids: set[int] | None = None,
    value_bin_support: torch.Tensor | None = None,
    collected_predictions: dict[int, list[ValidationFramePrediction]] | None = None,
) -> dict[str, float]:
    # Need this to prevent hanging
    _default_decoder_cache.clear()

    total_loss = 0.0
    total_acc = 0.0
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break

            batch = preprocess_batch(batch, preprocessor)
            model_batch = {
                k: v for k, v in batch.items() if isinstance(v, torch.Tensor)
            }

            loss, outputs = model(model_batch)
            value_logits = outputs["value_logits"]
            expected_value = outputs["expected_value"].squeeze(-1)

            batch_size = batch["target_bin"].shape[0]
            pred_bins = value_logits.argmax(dim=-1)
            acc = (pred_bins == batch["target_bin"]).float().mean()
            mae = torch.abs(expected_value - batch["target_value"]).mean()

            total_loss += float(loss.item()) * batch_size
            total_acc += float(acc.item()) * batch_size
            total_mae += float(mae.item()) * batch_size
            total_samples += batch_size

            if collected_predictions is not None:
                probs = outputs["value_probs"].detach()
                if value_bin_support is None:
                    support = torch.linspace(
                        -1.0,
                        0.0,
                        probs.shape[-1],
                        device=probs.device,
                        dtype=probs.dtype,
                    )
                else:
                    support = value_bin_support.to(
                        device=probs.device, dtype=probs.dtype
                    )
                reconstructed = (probs * support.unsqueeze(0)).sum(dim=-1)

                for idx in range(batch_size):
                    episode_index = int(batch["episode_index"][idx])
                    if (
                        collect_episode_ids is not None
                        and episode_index not in collect_episode_ids
                    ):
                        continue
                    prediction = ValidationFramePrediction(
                        frame_index=int(batch["frame_index"][idx]),
                        success=int(batch["success"][idx]),
                        target_value=float(batch["target_value"][idx]),
                        target_bin=int(batch["target_bin"][idx]),
                        predicted_bin=int(pred_bins[idx]),
                        reconstructed_value=float(reconstructed[idx]),
                        predicted_probs=probs[idx].float().cpu().numpy().copy(),
                    )
                    collected_predictions.setdefault(episode_index, []).append(
                        prediction
                    )

    if total_samples == 0:
        return {
            "loss": float("nan"),
            "bin_acc": float("nan"),
            "value_mae": float("nan"),
        }
    return {
        "loss": total_loss / total_samples,
        "bin_acc": total_acc / total_samples,
        "value_mae": total_mae / total_samples,
    }


def save_checkpoint(
    dest: Path,
    model: RECAPValueNetwork,
    preprocessor: Any,
    metrics: dict,
    cfg: RECAPValueTrainingConfig,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest)
    preprocessor.save_pretrained(dest, config_filename="policy_preprocessor.json")
    write_json(metrics, dest / "metrics.json")
    write_json(draccus.encode(cfg), dest / "train_config.json")


@parser.wrap()
def run_recap_value_train_val(cfg: RECAPValueTrainingConfig) -> None:
    """Train/validate RECAPValueNetwork with distributional bin supervision."""
    register_third_party_plugins()
    init_logging()
    set_seed(cfg.seed)

    output_dir = Path("outputs/value") / datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    write_json(draccus.encode(cfg), output_dir / "train_config.json")

    device = get_safe_torch_device(cfg.device, log=True)
    logging.info(f"Using device: {device}")

    wandb_run = None
    if cfg.wandb_project is not None:
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.job_name,
            config=draccus.encode(cfg),
        )
        logging.info(f"W&B run: {wandb_run.url}")

    dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
        vcodec="auto",
    )

    success_by_episode = load_episode_success_from_dataset(dataset)
    logging.info(
        f"Loaded success labels for {len(success_by_episode)} episodes "
        "from the dataset's 'success' column."
    )

    step_rewards = cfg.reward.compute_step_rewards(dataset=dataset, device=device)

    frame_targets = build_frame_targets(
        dataset=dataset,
        success_by_episode=success_by_episode,
        c_fail=cfg.c_fail,
        num_value_bins=cfg.model.num_value_bins,
        step_rewards=step_rewards,
    )
    train_targets, val_targets = split_train_val_targets(
        frame_targets=frame_targets,
        val_ratio=cfg.val_split_ratio,
        seed=cfg.seed,
    )

    preprocessor = build_value_preprocessor(
        dataset=dataset,
        tokenizer_name=cfg.model.text_backbone,
        model_precision=cfg.model.precision,
        device=str(device),
    )
    logging.info("Created pi05 preprocessor for value network training")

    train_dataset = RECAPFrameSupervisionDataset(
        base_dataset=dataset,
        frame_targets=train_targets,
    )
    val_dataset = RECAPFrameSupervisionDataset(
        base_dataset=dataset,
        frame_targets=val_targets,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    cfg.model.device = str(device)
    cfg.model.repo_id = cfg.value_repo_id
    cfg.model.push_to_hub = cfg.push_to_hub
    model = RECAPValueNetwork(cfg.model).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logging.info(
        f"Trainable parameters: {sum(p.numel() for p in trainable_params):,} "
        f"/ {sum(p.numel() for p in model.parameters()):,} total"
    )
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    pi05_defaults = PI05Config(optimizer_lr=cfg.learning_rate)
    optimizer_preset = pi05_defaults.get_optimizer_preset()
    scheduler_preset = pi05_defaults.get_scheduler_preset()
    max_grad_norm = optimizer_preset.grad_clip_norm

    optimizer = optimizer_preset.build(trainable_params)
    scheduler = scheduler_preset.build(optimizer, num_training_steps=cfg.train_steps)
    logging.info(
        "Using pi05 optimizer/scheduler presets: "
        f"lr={optimizer_preset.lr} betas={optimizer_preset.betas} "
        f"eps={optimizer_preset.eps} wd={optimizer_preset.weight_decay} "
        f"grad_clip={max_grad_norm} "
        f"warmup={scheduler_preset.num_warmup_steps} "
        f"decay={scheduler_preset.num_decay_steps} "
        f"decay_lr={scheduler_preset.decay_lr} "
        f"train_steps={cfg.train_steps}"
    )

    plot_episode_ids = select_validation_plot_episode_ids(
        frame_targets=val_targets,
        max_episodes=VAL_PLOT_NUM_EPISODES,
    )
    if plot_episode_ids:
        logging.info(f"Validation plots will track episodes: {plot_episode_ids}")
    plot_episode_id_set = set(plot_episode_ids)
    plot_targets = [
        target for target in val_targets if target.episode_index in plot_episode_id_set
    ]
    val_plot_loader: DataLoader | None = None
    if plot_targets:
        val_plot_dataset = RECAPFrameSupervisionDataset(
            base_dataset=dataset,
            frame_targets=plot_targets,
        )
        val_plot_loader = DataLoader(
            val_plot_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    logging.info(
        "Starting training: "
        f"{len(train_targets)} train frames / {len(val_targets)} val frames "
        f"from {len(set(t.episode_index for t in train_targets))} train episodes and "
        f"{len(set(t.episode_index for t in val_targets))} val episodes."
    )

    # Need this to prevent hanging
    _default_decoder_cache.clear()

    train_iter = cycle(train_loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    best_val_mae = float("inf")
    patience_counter = 0
    history: list[dict] = []
    should_stop = False

    start_time = time.perf_counter()
    last_log_step = 0
    last_log_time = start_time

    for global_step in range(1, cfg.train_steps + 1):
        batch = next(train_iter)
        train_metrics = train_step(
            model=model,
            batch=batch,
            preprocessor=preprocessor,
            optimizer=optimizer,
            scheduler=scheduler,
            max_grad_norm=max_grad_norm,
        )

        is_log_step = (
            global_step == 1
            or global_step % cfg.log_every_n_steps == 0
            or global_step == cfg.train_steps
        )
        if not is_log_step:
            continue

        # ─── Train log ──────────────────────────────────────────────
        now = time.perf_counter()
        elapsed = now - start_time
        window_elapsed = max(now - last_log_time, 1e-9)
        steps_per_sec = (global_step - last_log_step) / window_elapsed
        eta = max(cfg.train_steps - global_step, 0) / max(steps_per_sec, 1e-9)
        lr = optimizer.param_groups[0]["lr"]

        logging.info(
            f"[step {global_step}/{cfg.train_steps}] "
            f"loss={train_metrics['loss']:.5f} "
            f"acc={train_metrics['bin_acc']:.4f} "
            f"mae={train_metrics['value_mae']:.5f} "
            f"lr={lr:.2e} "
            f"it/s={steps_per_sec:.2f} "
            f"elapsed={format_duration(elapsed)} "
            f"eta={format_duration(eta)}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss": train_metrics["loss"],
                    "train/bin_acc": train_metrics["bin_acc"],
                    "train/value_mae": train_metrics["value_mae"],
                    "train/lr": lr,
                    "train/steps_per_sec": steps_per_sec,
                    "global_step": global_step,
                },
                step=global_step,
            )

        # ─── Validate ───────────────────────────────────────────────
        model.eval()
        try:
            val_metrics = validate(
                model=model,
                loader=val_loader,
                preprocessor=preprocessor,
                max_steps=cfg.max_val_steps,
                value_bin_support=model.value_bin_support,
            )
        except Exception as error:  # noqa: BLE001
            if not is_known_video_validation_error(error):
                model.train()
                raise
            logging.warning(
                f"[step {global_step}] Validation skipped due to persistent "
                f"video decoding/timestamp errors: {error}"
            )
            val_metrics = {
                "loss": float("nan"),
                "bin_acc": float("nan"),
                "value_mae": float("nan"),
            }
        else:
            logging.info(
                f"[step {global_step}/{cfg.train_steps}] "
                f"val_loss={val_metrics['loss']:.5f} "
                f"val_acc={val_metrics['bin_acc']:.4f} "
                f"val_mae={val_metrics['value_mae']:.5f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "val/loss": val_metrics["loss"],
                        "val/bin_acc": val_metrics["bin_acc"],
                        "val/value_mae": val_metrics["value_mae"],
                        "global_step": global_step,
                    },
                    step=global_step,
                )

        # ─── Plot pass (separate validation pass over plot episodes) ──
        do_plot = (
            cfg.plot_every_n_train_steps > 0
            and global_step % cfg.plot_every_n_train_steps == 0
            and bool(plot_episode_ids)
            and val_plot_loader is not None
        )
        if do_plot:
            collected_predictions: dict[int, list[ValidationFramePrediction]] = {}
            try:
                validate(
                    model=model,
                    loader=val_plot_loader,
                    preprocessor=preprocessor,
                    max_steps=None,
                    collect_episode_ids=plot_episode_id_set,
                    value_bin_support=model.value_bin_support,
                    collected_predictions=collected_predictions,
                )
            except Exception as error:  # noqa: BLE001
                if not is_known_video_validation_error(error):
                    model.train()
                    raise
                logging.warning(
                    f"[step {global_step}] Plot generation skipped due to "
                    f"video decode/timestamp issue: {error}"
                )
            else:
                plot_dir = output_dir / "validation_plots" / f"step_{global_step:08d}"
                saved_paths: list[Path] = []
                for episode_index in plot_episode_ids:
                    plot_path = plot_dir / f"episode_{episode_index:05d}.png"
                    did_save = save_validation_episode_plot(
                        dataset=dataset,
                        episode_index=episode_index,
                        predictions=collected_predictions.get(episode_index, []),
                        output_path=plot_path,
                        num_preview_frames=VAL_PLOT_NUM_FRAMES,
                        num_value_bins=cfg.model.num_value_bins,
                        image_size=cfg.model.image_size,
                    )
                    if did_save:
                        saved_paths.append(plot_path)
                if saved_paths:
                    logging.info(
                        f"[step {global_step}] Saved {len(saved_paths)} validation "
                        f"plot(s) under {plot_dir}"
                    )
                    if wandb_run is not None:
                        plot_images = {
                            f"val_plots/episode_{p.stem.split('_')[-1]}": (
                                wandb.Image(str(p))
                            )
                            for p in saved_paths
                        }
                        wandb_run.log(plot_images, step=global_step)

        model.train()

        # ─── History + checkpoints + early stop ─────────────────────
        saved_metrics = {
            "global_step": global_step,
            "train_loss": train_metrics["loss"],
            "train_bin_acc": train_metrics["bin_acc"],
            "train_value_mae": train_metrics["value_mae"],
            "val_loss": val_metrics["loss"],
            "val_bin_acc": val_metrics["bin_acc"],
            "val_value_mae": val_metrics["value_mae"],
            "lr": lr,
        }
        history.append(saved_metrics)
        write_json(history, output_dir / "metrics_history.json")  # ty: ignore[invalid-argument-type]

        is_save_step = (
            global_step % cfg.save_every_n_steps == 0 or global_step == cfg.train_steps
        )
        if is_save_step:
            save_checkpoint(
                checkpoints_dir / "last", model, preprocessor, saved_metrics, cfg
            )

        trigger_tag = f"step={global_step}/{cfg.train_steps}"
        val_mae = val_metrics["value_mae"]
        if not math.isnan(val_mae):
            if best_val_mae - val_mae > cfg.early_stopping_min_delta:
                best_val_mae = val_mae
                patience_counter = 0
                save_checkpoint(
                    checkpoints_dir / "best", model, preprocessor, saved_metrics, cfg
                )
                logging.info(
                    f"[{trigger_tag}] New best val_mae={val_mae:.5f}; "
                    "saved best checkpoint."
                )
            else:
                patience_counter += 1
                patience = cfg.early_stopping_patience
                if patience is not None:
                    logging.info(
                        f"[{trigger_tag}] No val_mae improvement "
                        f"({val_mae:.5f} vs best {best_val_mae:.5f}); "
                        f"patience {patience_counter}/{patience}"
                    )
                    if patience_counter >= patience:
                        logging.info(
                            f"[{trigger_tag}] Early stopping triggered after "
                            f"{patience_counter} validations without improvement."
                        )
                        should_stop = True

        # Reset window AFTER validation/plot/checkpoint so their wall time
        # is not counted as training throughput in the next window.
        last_log_step = global_step
        last_log_time = time.perf_counter()

        if should_stop:
            break

    logging.info(f"Training complete. Best val mae: {best_val_mae:.5f}")

    if cfg.push_to_hub and cfg.value_repo_id:
        from huggingface_hub import upload_file

        logging.info(f"Pushing value network to hub: {cfg.value_repo_id}")
        model.push_to_hub(cfg.value_repo_id)
        preprocessor.push_to_hub(cfg.value_repo_id)
        upload_file(
            path_or_fileobj=str(output_dir / "train_config.json"),
            path_in_repo="train_config.json",
            repo_id=cfg.value_repo_id,
            repo_type="model",
        )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    run_recap_value_train_val()  # ty: ignore[missing-argument]
