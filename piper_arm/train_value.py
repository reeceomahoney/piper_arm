"""Train a distributional value function from rollout datasets.

Uses cross-entropy loss over discretized return bins (RECAP-style).
Ground truth returns are computed analytically from steps_remaining + success.

Usage:
    python -m piper_arm.train_value --config_path configs/value.yaml
"""

import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import torch
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import cycle
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import batch_to_transition
from lerobot.processor.pipeline import EnvTransition, TransitionKey
from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME
from torch.utils.data import DataLoader

from piper_arm.value_model import ValueConfig, ValueModel


@dataclass
class TrainValueConfig:
    dataset_repo_id: str = "reece-omahoney/libero"
    dataset_root: str | None = None
    pretrained_path: str = "reece-omahoney/smolvla-libero-16-chunk"
    c_fail: float = 1000.0
    reward_type: str = "steps_remaining"  # "steps_remaining" or "maha_distance"

    value: ValueConfig = field(default_factory=ValueConfig)

    # Resume from a previous value-model checkpoint (.pt file)
    value_pretrained_path: str | None = None

    # Training
    batch_size: int = 32
    total_steps: int = 100_000

    # Logging & checkpointing
    log_interval: int = 100
    save_interval: int = 10_000
    output_dir: str | None = None
    wandb_project: str | None = "distal-value"
    wandb_run_name: str | None = None

    num_workers: int = 4
    seed: int = 42


def compute_returns(
    steps_remaining: torch.Tensor,
    success: torch.Tensor,
    max_episode_length: int,
    c_fail: float,
) -> torch.Tensor:
    """Compute ground-truth returns from episode metadata.

    Successful: G_t = -steps_remaining / max_episode_length
    Failed:     G_t = (-steps_remaining - c_fail + 1) / max_len, clipped to -1

    Args:
        steps_remaining: (B,) or (B, 1) int tensor.
        success: (B,) or (B, 1) bool tensor.
        max_episode_length: max steps in any episode.
        c_fail: failure penalty.

    Returns:
        (B,) float tensor with returns in [-1, 0].
    """
    steps = steps_remaining.float().squeeze(-1)
    succ = success.squeeze(-1)

    returns = torch.where(
        succ,
        -steps / max_episode_length,
        (-steps - c_fail + 1) / max_episode_length,
    )
    return returns.clamp(-1.0, 0.0)


def precompute_maha_returns(
    dataset: LeRobotDataset,
    max_episode_length: int,
) -> tuple[np.ndarray, float]:
    """Pre-compute per-frame returns using negative Mahalanobis distance as reward.

    For each episode, G_t = sum_{k=t}^{T-1} -maha[k]. Returns are normalized
    to [-1, 0] by dividing by the max absolute return across all frames.

    Args:
        dataset: LeRobot dataset with a ``maha_distance`` column.
        max_episode_length: max steps in any episode (unused but kept for API
            consistency; normalization is data-driven).

    Returns:
        (returns, norm_constant) where *returns* is a ``(num_frames,)`` float64
        array of normalized returns in ``[-1, 0]`` and *norm_constant* is the
        divisor used for normalization.
    """
    maha = np.array(
        [s.item() for s in dataset.hf_dataset["maha_distance"]], dtype=np.float64
    )
    episode_index = np.array(
        [s.item() for s in dataset.hf_dataset["episode_index"]], dtype=np.float64
    )

    num_frames = len(maha)
    returns = np.zeros(num_frames, dtype=np.float64)

    # Reverse cumulative sum of -maha within each episode
    i = num_frames - 1
    while i >= 0:
        ep = episode_index[i]
        cumsum = 0.0
        j = i
        # Find all frames in this episode (they are contiguous)
        while j >= 0 and episode_index[j] == ep:
            j -= 1
        ep_start = j + 1
        # Forward pass computing reverse cumsum
        cumsum = 0.0
        for k in range(i, ep_start - 1, -1):
            cumsum += -maha[k]
            returns[k] = cumsum
        i = ep_start - 1

    # Normalize to [-1, 0]
    norm_constant = np.abs(returns).max()
    if norm_constant > 0:
        returns = returns / norm_constant

    return returns, float(norm_constant)


VALUE_EXTRA_KEYS = ("steps_remaining", "success", "maha_distance", "index")


def batch_to_transition_with_extras(batch: dict[str, Any]) -> EnvTransition:
    """Wrap batch_to_transition to preserve value-training keys.

    The default converter only extracts known keys (task, index, etc.) into
    complementary_data. This version also preserves steps_remaining and success.
    """
    transition = batch_to_transition(batch)
    comp = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
    for key in VALUE_EXTRA_KEYS:
        if key in batch and key not in comp:
            comp[key] = batch[key]
    transition[TransitionKey.COMPLEMENTARY_DATA] = comp
    return transition


def load_value_preprocessor(pretrained_path: str):
    """Load preprocessor from a pretrained SmolVLA checkpoint.

    Image preprocessing and state padding are handled inside ValueModel.forward.
    """
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        to_transition=batch_to_transition_with_extras,  # type: ignore[invalid-argument-type]
    )
    return preprocessor


@draccus.wrap()
def main(cfg: TrainValueConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    now = datetime.now()
    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = (
            Path("outputs/value") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B ──
    if cfg.wandb_project:
        import wandb

        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=vars(cfg),
            dir=str(output_dir),
        )

    # ── Dataset ──
    ds_kwargs: dict = {"repo_id": cfg.dataset_repo_id}
    if cfg.dataset_root:
        ds_kwargs["root"] = cfg.dataset_root
    dataset = LeRobotDataset(**ds_kwargs)

    # Auto-detect max episode length from steps_remaining
    all_steps = dataset.hf_dataset["steps_remaining"]
    max_episode_length = max(s.item() for s in all_steps) + 1
    print(f"Max episode length: {max_episode_length}")
    print(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Model & preprocessor ──
    model = ValueModel(cfg.value)
    model = model.to(device)
    preprocessor = load_value_preprocessor(cfg.pretrained_path)

    # ── Optimizer & scheduler (SmolVLA presets) ──
    smolvla_cfg = SmolVLAConfig()
    optimizer_cfg = smolvla_cfg.get_optimizer_preset()
    scheduler_cfg = smolvla_cfg.get_scheduler_preset()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_cfg.build(params)
    scheduler = scheduler_cfg.build(optimizer, num_training_steps=cfg.total_steps)

    # ── Load pretrained value checkpoint ──
    if cfg.value_pretrained_path:
        ckpt = torch.load(
            cfg.value_pretrained_path, map_location=device, weights_only=False
        )
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded pretrained value model from {cfg.value_pretrained_path}")

    # ── Pre-compute maha returns if needed ──
    maha_norm = None
    maha_returns_tensor = None
    if cfg.reward_type == "maha_distance":
        print("Pre-computing Mahalanobis distance returns...")
        maha_returns_arr, maha_norm = precompute_maha_returns(
            dataset, max_episode_length
        )
        maha_returns_tensor = torch.tensor(
            maha_returns_arr, dtype=torch.float32, device=device
        )
        print(f"Maha norm constant: {maha_norm:.4f}")

    # ── Training loop ──
    model.train()
    data_iter = cycle(loader)

    for step in range(1, cfg.total_steps + 1):
        batch = next(data_iter)
        batch = preprocessor(batch)

        if cfg.reward_type == "maha_distance":
            assert maha_returns_tensor is not None
            indices = batch["index"].long().to(device)
            returns = maha_returns_tensor[indices]
        else:
            returns = compute_returns(
                batch["steps_remaining"],
                batch["success"],
                max_episode_length,
                cfg.c_fail,
            )

        logits = model(batch)
        targets = ValueModel.returns_to_bins(returns, cfg.value.n_bins)
        loss = F.cross_entropy(logits, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, optimizer_cfg.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # ── Logging ──
        if step % cfg.log_interval == 0:
            with torch.no_grad():
                pred_values = model.predict_value(logits)
                mae = (pred_values - returns).abs().mean().item()

            log = {
                "loss": loss.item(),
                "mae": mae,
                "lr": scheduler.get_last_lr()[0],
                "step": step,
            }
            lr_str = f"{log['lr']:.2e}"
            print(
                f"[step {step:>6d}] loss={log['loss']:.4f}"
                f"  mae={log['mae']:.4f}  lr={lr_str}"
            )
            if cfg.wandb_project:
                wandb.log(log, step=step)

        # ── Checkpointing ──
        if step % cfg.save_interval == 0:
            ckpt_path = output_dir / f"checkpoint_{step}.pt"
            ckpt_data: dict[str, Any] = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
            }
            if maha_norm is not None:
                ckpt_data["maha_norm"] = maha_norm
            torch.save(ckpt_data, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Save final checkpoint
    final_path = output_dir / "checkpoint_final.pt"
    final_ckpt_data: dict[str, Any] = {
        "step": cfg.total_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
    }
    if maha_norm is not None:
        final_ckpt_data["maha_norm"] = maha_norm
    torch.save(final_ckpt_data, final_path)
    print(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
