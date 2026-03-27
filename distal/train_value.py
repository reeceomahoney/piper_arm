"""Train a distributional value function from rollout datasets.

Uses cross-entropy loss over discretized return bins (RECAP-style).
Ground truth returns are computed from shared reward helpers.
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
from lerobot_policy_advantage.configuration_advantage import AdvantageConfig
from lerobot_policy_advantage.processor_advantage import (
    make_advantage_pre_post_processors,
)
from torch.utils.data import DataLoader

from distal.rewards import build_reward_context, save_reward_context
from distal.value_model import ValueConfig, ValueFunction


@dataclass
class TrainValueConfig:
    dataset_repo_id: str = "reece-omahoney/libero-10"
    gamma: float = 1.0
    reward_type: str = "steps"  # "steps" or "maha"
    failure_penalty_scale: float = 0.125
    stats_repo_id: str = "reece-omahoney/maha-stats"
    base_policy: str = "reece-omahoney/adv-libero-base"

    value: ValueConfig = field(default_factory=ValueConfig)
    value_repo_id: str = "reece-omahoney/value-success-eighth-eplen"
    push_to_hub: bool = True

    # Training
    batch_size: int = 32
    total_steps: int = 100_000

    # Logging & checkpointing
    log_interval: int = 100
    save_interval: int = 10_000
    output_dir: str | None = None
    wandb_project: str | None = "distal-value"

    num_workers: int = 4
    seed: int = 42
    use_amp: bool = True


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
            name=cfg.value_repo_id.split("/")[-1],
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
    success_arr = np.array([bool(s.item()) for s in dataset.hf_dataset["success"]])
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
    reward_context = build_reward_context(
        cfg,
        dataset=dataset,
        device=str(device),
        episode_index=episode_index,
        success=success_arr,
        steps_remaining=steps_remaining_arr,
        max_episode_length=max_episode_length,
    )
    print(f"Reward norm constant: {reward_context.normalization_constant:.4f}")
    print(f"Failure penalty: {reward_context.failure_penalty:.4f}")

    returns_tensor = torch.tensor(
        reward_context.returns, dtype=torch.float32, device=device
    )

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

        indices = batch["index"].long().to(device)
        batch = preprocessor(batch)
        batch["returns"] = returns_tensor[indices]

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
            save_reward_context(ckpt_dir, reward_context, dataset.num_frames)
            print(f"Saved checkpoint: {ckpt_dir}")

    # Save final checkpoint
    final_dir = output_dir / "checkpoint_final"
    model.save_pretrained(final_dir)
    save_reward_context(final_dir, reward_context, dataset.num_frames)
    print(f"Training complete. Final checkpoint: {final_dir}")

    # Push to hub
    if cfg.push_to_hub:
        model.push_to_hub(cfg.value_repo_id)
        print(f"Pushed to https://huggingface.co/{cfg.value_repo_id}")


if __name__ == "__main__":
    main()
