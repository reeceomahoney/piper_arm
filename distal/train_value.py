"""Train a distributional value function from rollout datasets.

Uses cross-entropy loss over discretized return bins (RECAP-style).
Ground truth returns are computed analytically from steps_remaining + success.
"""

import random
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import draccus
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import cycle
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from torch.utils.data import DataLoader

from distal.value_model import ValueConfig, ValueFunction
from lerobot_policy_advantage.configuration_advantage import AdvantageConfig
from lerobot_policy_advantage.processor_advantage import (
    make_advantage_pre_post_processors,
)


@dataclass
class TrainValueConfig:
    dataset_repo_id: str = "reece-omahoney/libero-10"
    c_fail: float = 1000.0
    reward_type: str = "steps_remaining"  # "steps_remaining" or "maha_distance"
    stats_repo_id: str = "reece-omahoney/maha-stats"
    base_policy: str = "reece-omahoney/adv-libero-base"

    value: ValueConfig = field(default_factory=ValueConfig)
    value_repo_id: str = "reece-omahoney/value-success-expert"
    push_to_hub: bool = False

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
    use_amp: bool = True


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
    maha: np.ndarray,
    episode_index: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Pre-compute per-frame returns using negative Mahalanobis distance as reward.

    For each episode, G_t = sum_{k=t}^{T-1} -maha[k]. Returns are normalized
    to [-1, 0] by dividing by the max absolute return across all frames.

    Args:
        maha: (N,) float64 array of per-frame Mahalanobis distances.
        episode_index: (N,) int array of episode indices.

    Returns:
        (returns, norm_constant) where *returns* is a ``(num_frames,)`` float64
        array of normalized returns in ``[-1, 0]`` and *norm_constant* is the
        divisor used for normalization.
    """

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
    dataset = LeRobotDataset(cfg.dataset_repo_id)

    frame_index = np.array([s.item() for s in dataset.hf_dataset["frame_index"]])
    episode_index = np.array([s.item() for s in dataset.hf_dataset["episode_index"]])
    ep_lengths = Counter(episode_index.tolist())
    steps_remaining_arr = np.array(
        [ep_lengths[ep] - fi - 1 for ep, fi in zip(episode_index, frame_index)],
        dtype=np.int32,
    )
    max_episode_length = max(ep_lengths.values())
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
    if cfg.value.pretrained_path:
        model = ValueFunction.from_pretrained(str(cfg.value.pretrained_path))
        print(f"Loaded pretrained value model from {cfg.value.pretrained_path}")
    else:
        model = ValueFunction(cfg.value)
    model = model.to(device)
    preprocessor, _ = make_advantage_pre_post_processors(AdvantageConfig())

    # ── Optimizer & scheduler ──
    optimizer_cfg = cfg.value.get_optimizer_preset()
    scheduler_cfg = cfg.value.get_scheduler_preset()
    assert isinstance(scheduler_cfg, CosineDecayWithWarmupSchedulerConfig)
    scheduler_cfg.num_decay_steps = cfg.total_steps

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_cfg.build(params)
    assert isinstance(optimizer, torch.optim.Optimizer)
    scheduler = scheduler_cfg.build(optimizer, num_training_steps=cfg.total_steps)
    assert scheduler is not None

    # ── Pre-compute returns tensors ──
    steps_remaining_tensor = torch.tensor(
        steps_remaining_arr, dtype=torch.float32, device=device
    )
    maha_norm = None
    maha_returns_tensor = None
    if cfg.reward_type == "maha_distance":
        from huggingface_hub import hf_hub_download
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_policy
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        from distal.compute_maha_stats import compute_maha_distances

        stats_file = hf_hub_download(cfg.stats_repo_id, "stats.npz")
        data = np.load(stats_file)
        gauss_mean = data["mean"]
        gauss_cov_inv = data["cov_inv"]
        print(
            f"Loaded Gaussian stats from {cfg.stats_repo_id}, dim={gauss_mean.shape[0]}"
        )

        print("Loading policy for Mahalanobis distance computation...")
        policy_cfg = PreTrainedConfig.from_pretrained(cfg.base_policy)
        policy_cfg.pretrained_path = Path(cfg.base_policy)
        policy_cfg.device = str(device)
        maha_policy = make_policy(cfg=policy_cfg)
        assert isinstance(maha_policy, (PI05Policy, SmolVLAPolicy))
        maha_policy.eval()
        policy_preprocessor, _ = make_advantage_pre_post_processors(AdvantageConfig())

        maha_arr = compute_maha_distances(
            maha_policy,
            policy_preprocessor,
            dataset,
            gauss_mean,
            gauss_cov_inv,
            cfg.batch_size,
            cfg.num_workers,
        )
        maha_returns_arr, maha_norm = precompute_maha_returns(maha_arr, episode_index)
        maha_returns_tensor = torch.tensor(
            maha_returns_arr, dtype=torch.float32, device=device
        )
        print(f"Maha norm constant: {maha_norm:.4f}")
        model.get_buffer("maha_norm").fill_(maha_norm)

    # ── Training loop ──
    model.train()
    data_iter = cycle(loader)
    autocast = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if cfg.use_amp
        else nullcontext()
    )

    for step in range(1, cfg.total_steps + 1):
        dataloading_start = time.perf_counter()
        batch = next(data_iter)
        dataloading_s = time.perf_counter() - dataloading_start

        success = batch.get("success")
        indices = batch["index"].long().to(device)
        batch = preprocessor(batch)

        if cfg.reward_type == "maha_distance":
            assert maha_returns_tensor is not None
            returns = maha_returns_tensor[indices]
        else:
            returns = compute_returns(
                steps_remaining_tensor[indices],
                success.to(device),
                max_episode_length,
                cfg.c_fail,
            )

        batch["returns"] = returns

        update_start = time.perf_counter()
        with autocast:
            loss, info = model(batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, optimizer_cfg.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        update_s = time.perf_counter() - update_start

        # ── Logging ──
        if step % cfg.log_interval == 0:
            log = {
                "loss": loss.item(),
                "mae": info["mae"],
                "lr": scheduler.get_last_lr()[0],
                "update_s": update_s,
                "dataloading_s": dataloading_s,
                "step": step,
            }
            lr_str = f"{log['lr']:.2e}"
            print(
                f"[step {step:>6d}] loss={log['loss']:.4f}"
                f"  mae={log['mae']:.4f}  lr={lr_str}"
                f"  update_s={update_s:.3f}  dataloading_s={dataloading_s:.3f}"
            )
            if cfg.wandb_project:
                wandb.log(log, step=step)

        # ── Checkpointing ──
        if step % cfg.save_interval == 0:
            ckpt_dir = output_dir / f"checkpoint_{step}"
            model.save_pretrained(ckpt_dir)
            print(f"Saved checkpoint: {ckpt_dir}")

    # Save final checkpoint
    final_dir = output_dir / "checkpoint_final"
    model.save_pretrained(final_dir)
    print(f"Training complete. Final checkpoint: {final_dir}")

    # Push to hub
    if cfg.push_to_hub:
        model.push_to_hub(cfg.value_repo_id)
        print(f"Pushed to https://huggingface.co/{cfg.value_repo_id}")


if __name__ == "__main__":
    main()
