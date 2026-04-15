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
from torch.utils.data import DataLoader, Subset

from distal.rewards import build_reward_context, save_reward_context
from distal.value_model import ValueConfig, ValueFunction


@dataclass
class TrainValueConfig:
    dataset_repo_id: str = "reece-omahoney/libero-10"
    gamma: float = 1.0
    reward_type: str = "steps"  # "steps" or "maha"
    failure_penalty_scale: float = 1.0
    stats_repo_id: str = "reece-omahoney/maha-stats"

    value: ValueConfig = field(default_factory=ValueConfig)
    value_repo_id: str = "reece-omahoney/value-steps-gemma"
    push_to_hub: bool = True

    # Training
    batch_size: int = 32
    total_steps: int = 20_000

    # Logging & checkpointing
    log_interval: int = 100
    val_interval: int = 100
    max_val_batches: int = 20
    plot_interval: int = 500
    save_interval: int = 500
    output_dir: str | None = None
    wandb_project: str | None = "distal-value"

    val_fraction: float = 0.1
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

    all_episodes = sorted(ep_lengths.keys())
    n_val = max(1, int(len(all_episodes) * cfg.val_fraction))
    rng = random.Random(cfg.seed)
    val_episodes = set(rng.sample(all_episodes, n_val))
    train_episodes = set(all_episodes) - val_episodes

    train_indices = [i for i, ep in enumerate(episode_index) if ep in train_episodes]
    val_indices = [i for i, ep in enumerate(episode_index) if ep in val_episodes]
    episode_to_indices: dict[int, list[int]] = {}
    for i, ep in enumerate(episode_index):
        episode_to_indices.setdefault(int(ep), []).append(i)
    for ep in episode_to_indices:
        episode_to_indices[ep].sort(key=lambda i: int(frame_index[i]))
    plot_rng = random.Random(cfg.seed + 1)
    print(
        f"Train: {len(train_episodes)} episodes ({len(train_indices)} frames)  "
        f"Val: {len(val_episodes)} episodes ({len(val_indices)} frames)"
    )

    loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ── Model & preprocessor ──
    model = ValueFunction(cfg.value)
    model = model.to(device)

    # Use pi05's preprocessor so state is QUANTILES-normalized to [-1, 1]
    # before Pi05PrepareStateTokenizerProcessorStep discretizes it into the
    # text prompt (matches recap_train_value_network.py:_build_preprocessor).
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.feature_utils import dataset_to_policy_features
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

    features = dataset_to_policy_features(dataset.meta.features)
    output_features = {
        k: f for k, f in features.items() if f.type is FeatureType.ACTION
    }
    input_features = {k: f for k, f in features.items() if k not in output_features}
    policy_cfg = PI05Config(
        input_features=input_features,
        output_features=output_features,
        paligemma_variant=cfg.value.paligemma_variant,
        device=str(device),
    )
    raw_stats = dataset.meta.stats or {}
    dataset_stats = {
        k: {sk: torch.as_tensor(sv) for sk, sv in v.items()}
        for k, v in raw_stats.items()
    }
    preprocessor, _ = make_pi05_pre_post_processors(
        config=policy_cfg,
        dataset_stats=dataset_stats,
    )

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
        if step % cfg.val_interval == 0:
            do_plot = bool(cfg.wandb_project) and step % cfg.plot_interval == 0
            plot_ep_data: dict[int, list[tuple[int, float, float]]] = {}
            plot_ep_ids: list[int] = []
            if do_plot:
                val_window_size = cfg.max_val_batches * cfg.batch_size
                window_eps = sorted(
                    {int(episode_index[i]) for i in val_indices[:val_window_size]}
                )
                n_plot = min(4, len(window_eps))
                plot_ep_ids = plot_rng.sample(window_eps, n_plot)
                plot_ep_data = {ep: [] for ep in plot_ep_ids}

            model.eval()
            val_mae_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for val_step, val_batch in enumerate(val_loader):
                    if val_step >= cfg.max_val_batches:
                        break
                    val_indices_batch = val_batch["index"].long().to(device)
                    val_batch = preprocessor(val_batch)
                    val_batch["returns"] = returns_tensor[val_indices_batch]
                    with autocast:
                        val_logits = model.compute_logits(val_batch)
                    val_preds = model.logits_to_value(val_logits)
                    val_mae_sum += (val_preds - val_batch["returns"]).abs().sum().item()
                    val_count += val_preds.shape[0]

                    if do_plot:
                        idx_cpu = val_indices_batch.cpu().numpy()
                        preds_cpu = val_preds.float().cpu().numpy()
                        gts_cpu = val_batch["returns"].float().cpu().numpy()
                        for j, gi in enumerate(idx_cpu):
                            ep = int(episode_index[gi])
                            if ep in plot_ep_data:
                                plot_ep_data[ep].append(
                                    (
                                        int(frame_index[gi]),
                                        float(preds_cpu[j]),
                                        float(gts_cpu[j]),
                                    )
                                )
            val_mae = val_mae_sum / val_count
            model.train()

            log = {
                "loss": loss.item(),
                "mae": info["mae"],
                "val_mae": val_mae,
                "lr": scheduler.get_last_lr()[0],
                "update_s": update_s,
                "dataloading_s": dataloading_s,
                "step": step,
            }
            lr_str = f"{log['lr']:.2e}"
            print(
                f"[step {step:>6d}] loss={log['loss']:.4f}"
                f"  mae={log['mae']:.4f}  val_mae={val_mae:.4f}  lr={lr_str}"
                f"  update_s={update_s:.3f}  dataloading_s={dataloading_s:.3f}"
            )
            if cfg.wandb_project:
                wandb.log(log, step=step)

            if do_plot:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(2, 2, figsize=(10, 7))
                axes_flat = axes.flatten()
                for ax_idx, ep_id in enumerate(plot_ep_ids):
                    rows = sorted(plot_ep_data[ep_id], key=lambda r: r[0])
                    frames = [r[0] for r in rows]
                    preds_arr = [r[1] for r in rows]
                    gts_arr = [r[2] for r in rows]
                    ax = axes_flat[ax_idx]
                    ax.plot(frames, gts_arr, label="ground truth", linewidth=2)
                    ax.plot(frames, preds_arr, label="predicted", linewidth=2)
                    ax.set_xlabel("frame")
                    ax.set_ylabel("return")
                    ax.set_title(f"episode {ep_id}")
                    ax.legend()
                    ax.grid(alpha=0.3)
                for ax_idx in range(len(plot_ep_ids), len(axes_flat)):
                    axes_flat[ax_idx].axis("off")
                fig.suptitle(f"step {step}")
                fig.tight_layout()
                wandb.log({"episode_value": wandb.Image(fig)}, step=step)
                plt.close(fig)

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
        from huggingface_hub import upload_file

        from distal.rewards import REWARD_CONTEXT_FILENAME

        model.push_to_hub(cfg.value_repo_id)

        reward_context_path = final_dir / REWARD_CONTEXT_FILENAME
        upload_file(
            path_or_fileobj=str(reward_context_path),
            path_in_repo=REWARD_CONTEXT_FILENAME,
            repo_id=cfg.value_repo_id,
        )
        print(f"Pushed to https://huggingface.co/{cfg.value_repo_id}")


if __name__ == "__main__":
    main()
