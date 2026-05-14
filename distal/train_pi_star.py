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
"""Train/val script for the PiStar06 advantage-conditioned Pi0.5 policy.

Standalone training loop with an episode-level train/val split and validation
metrics that measure how well the model differentiates positive vs negative
advantage distributions.

Advantages are **pre-computed** before the training loop starts using a frozen
Pi0.5-based RECAPValueNetwork (PaliGemma backbone).  The value network runs
once over the full dataset, then the advantage for each frame is injected into
training/validation batches via a lookup dict keyed by absolute frame index.
"""

import gc
import logging
import resource
import time as time_module
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import draccus
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, broadcast_object_list
from lerobot.common.train_utils import save_training_state
from lerobot.configs import parser
from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.utils.io_utils import write_json
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import cycle
from lerobot_policy_pistar06.configuration_pistar06 import PiStar06Config
from lerobot_policy_pistar06.modeling_pistar06 import PiStar06Policy
from torch.utils.data import DataLoader

from distal.advantages import (
    AdvantageConfig,
    inject_advantages,
    load_vn_metadata,
    prepare_advantages,
)
from distal.sim_eval import (
    LiberoEvalConfig,
    LiberoPlusEvalConfig,
    run_libero_eval,
    run_libero_plus_eval,
)
from distal.train_value import (
    build_frame_targets,
    format_duration,
    is_known_video_validation_error,
    load_episode_success_from_dataset,
    split_train_val_targets,
)


@dataclass
class RECAPPiStarTrainingConfig:
    """Configuration for RECAP PiStar06 advantage-conditioned Pi0.5 policy training."""

    job_name: str = "pistar-knn-rel-libero-plus"
    dataset_repo_id: str = "reece-omahoney/pi05-libero-plus"

    train_steps: int = 20_000
    batch_size: int = 64
    num_workers: int = 4
    val_split_ratio: float = 0.1
    seed: int = 42
    device: str = "cuda"
    log_every_n_steps: int = 100
    max_val_steps: int | None = 50
    sim_eval_every_n_train_steps: int = 500

    policy: PiStar06Config = field(
        default_factory=lambda: PiStar06Config(
            pretrained_path=Path("lerobot/pi05-libero"),
            dtype="bfloat16",
            n_action_steps=10,
            gradient_checkpointing=True,
            compile_model=True,
        )
    )
    advantage: AdvantageConfig = field(default_factory=AdvantageConfig)

    # Sim eval
    eval_cfg: LiberoEvalConfig | LiberoPlusEvalConfig = field(
        # default_factory=lambda: LiberoEvalConfig(
        #     suites=["libero_10"],
        #     task_ids=[8],
        #     n_envs_per_task=25,
        #     n_episodes_per_task=2,
        # )
        default_factory=lambda: LiberoPlusEvalConfig(
            base_task="turn_on_the_stove", parallel_envs=25
        )
    )

    # Hub push for trained policy
    push_to_hub: bool = True

    # Weights & Biases (optional; set wandb_project to enable)
    wandb_project: str | None = "distal"
    wandb_entity: str | None = None


def _log_memory(label: str) -> None:
    """Log CPU RSS and CUDA memory at a named checkpoint."""
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    parts = [f"[MEM {label}] CPU_RSS={rss_mb:.0f}MB"]
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        parts.append(f"CUDA_alloc={alloc:.0f}MB CUDA_reserved={reserved:.0f}MB")
    logging.info(" ".join(parts))


def _init_wandb(cfg: RECAPPiStarTrainingConfig):
    """Initialise a W&B run if ``wandb_project`` is set, otherwise return ``None``."""
    if cfg.wandb_project is None:
        return None
    import os

    import wandb

    # Under a sweep agent, let wandb auto-generate a unique trial name
    # rather than collapsing every trial to the same `name`.
    in_sweep = "WANDB_SWEEP_ID" in os.environ
    wandb_name = None if (in_sweep or not cfg.job_name) else cfg.job_name

    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=wandb_name,
        config=asdict(cfg),
    )
    logging.info(f"W&B run: {run.url}")
    return run


def _resolve_policy_config(
    cfg: RECAPPiStarTrainingConfig,
    train_dataset: LeRobotDataset,
    c_fail: float,
) -> PiStar06Config:
    """Inject runtime-resolved fields onto cfg.policy and return it.

    Static knobs (architecture, dtype, advantage_dropout, cfg_beta, …) come
    from cfg.policy via CLI overrides; this helper only sets the fields that
    can't be defaulted (input/output features from the dataset, c_fail from
    the value network).
    """
    features = dataset_to_policy_features(train_dataset.meta.features)
    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }
    cfg.policy.input_features = input_features
    cfg.policy.output_features = output_features
    cfg.policy.c_fail = c_fail
    policy_cfg = cfg.policy
    return policy_cfg


# ── Validation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def _run_validation(
    policy,
    loader: DataLoader,
    preprocessor,
    advantage_lookup: dict[int, float],
    success_by_episode: dict[int, int],
    device: torch.device,
    max_steps: int | None = None,
) -> dict[str, float]:
    """Two-pass validation computing stratified loss and conditioning accuracy.

    For each batch:
      Pass 1 -- correct advantage embedding  -> per-sample loss_correct
      Pass 2 -- flipped advantage embedding  -> per-sample loss_wrong
    Both passes share identical noise and flow time for fair comparison.
    """
    policy.eval()

    total_loss = 0.0
    total_loss_pos = 0.0
    total_loss_neg = 0.0
    total_n_pos = 0
    total_n_neg = 0
    total_correct_wins = 0.0
    total_gap = 0.0
    total_gap_pos = 0.0
    total_gap_neg = 0.0
    total_samples = 0

    total_aligned = 0.0
    total_aligned_success = 0.0
    total_aligned_failure = 0.0
    total_success_samples = 0
    total_failure_samples = 0

    cap = f"/{max_steps}" if max_steps is not None else ""
    logging.info(f"Running validation (max_steps{cap})...")
    val_iter = iter(loader)
    step = -1
    skipped = 0
    while True:
        try:
            batch = next(val_iter)
        except StopIteration:
            break
        except RuntimeError as exc:
            if not is_known_video_validation_error(exc):
                raise
            skipped += 1
            logging.warning(
                f"Skipping validation batch due to video decode error "
                f"(skipped {skipped} so far): {exc}"
            )
            continue
        step += 1

        if max_steps is not None and step >= max_steps:
            break

        batch = preprocessor(batch)
        batch = inject_advantages(batch, advantage_lookup, device)

        advantages, _ = policy._compute_advantages(batch)
        true_indicator = advantages > policy.config.advantage_threshold
        pos_mask = true_indicator
        neg_mask = ~true_indicator

        B = batch[OBS_LANGUAGE_TOKENS].shape[0]

        ep_indices = batch["episode_index"]
        episode_success = torch.tensor(
            [success_by_episode[int(idx)] for idx in ep_indices],
            device=device,
            dtype=torch.bool,
        )
        adv_positive = advantages > policy.config.advantage_threshold
        aligned = (adv_positive == episode_success).float()
        total_aligned += aligned.sum().item()

        success_mask = episode_success
        failure_mask = ~episode_success
        n_success = success_mask.sum().item()
        n_failure = failure_mask.sum().item()
        total_success_samples += n_success
        total_failure_samples += n_failure
        if n_success > 0:
            total_aligned_success += aligned[success_mask].sum().item()
        if n_failure > 0:
            total_aligned_failure += aligned[failure_mask].sum().item()

        padded_actions = policy.prepare_action(batch)
        noise = torch.randn_like(padded_actions)
        fm_time = torch.rand(B, device=device)

        with (
            torch.no_grad(),
            torch.autocast(device_type=device.type, dtype=torch.bfloat16),
        ):
            # Pass 1: correct advantage embedding (no dropout)
            losses_correct = policy._forward_with_advantage(
                batch, true_indicator, dropout_mask=None, noise=noise, time=fm_time
            )
            original_action_dim = policy.config.output_features[ACTION].shape[0]
            losses_correct = losses_correct[:, :, :original_action_dim]
            actions_is_pad = batch.get("action_is_pad")
            if actions_is_pad is not None:
                losses_correct = losses_correct * (~actions_is_pad).unsqueeze(-1)
            loss_correct = losses_correct.mean(dim=(1, 2))

            # Pass 2: flipped advantage embedding (no dropout)
            losses_wrong = policy._forward_with_advantage(
                batch, ~true_indicator, dropout_mask=None, noise=noise, time=fm_time
            )
            losses_wrong = losses_wrong[:, :, :original_action_dim]
            if actions_is_pad is not None:
                losses_wrong = losses_wrong * (~actions_is_pad).unsqueeze(-1)
            loss_wrong = losses_wrong.mean(dim=(1, 2))

        total_loss += loss_correct.sum().item()
        n_pos = pos_mask.sum().item()
        n_neg = neg_mask.sum().item()
        total_n_pos += n_pos
        total_n_neg += n_neg
        if n_pos > 0:
            total_loss_pos += loss_correct[pos_mask].sum().item()
        if n_neg > 0:
            total_loss_neg += loss_correct[neg_mask].sum().item()

        correct_wins = (loss_correct < loss_wrong).float()
        total_correct_wins += correct_wins.sum().item()

        gap = loss_wrong - loss_correct
        total_gap += gap.sum().item()
        if n_pos > 0:
            total_gap_pos += gap[pos_mask].sum().item()
        if n_neg > 0:
            total_gap_neg += gap[neg_mask].sum().item()

        total_samples += B

    if skipped > 0:
        logging.warning(
            f"Validation skipped {skipped} batches due to video decode errors"
        )

    if total_samples == 0:
        return {
            "val_loss": float("nan"),
            "val_loss_pos": float("nan"),
            "val_loss_neg": float("nan"),
            "val_n_pos": 0,
            "val_n_neg": 0,
            "val_conditioning_accuracy": float("nan"),
            "val_conditioning_gap": float("nan"),
            "val_conditioning_gap_pos": float("nan"),
            "val_conditioning_gap_neg": float("nan"),
            "val_adv_episode_alignment": float("nan"),
            "val_alignment_on_success": float("nan"),
            "val_alignment_on_failure": float("nan"),
        }

    return {
        "val_loss": total_loss / total_samples,
        "val_loss_pos": total_loss_pos / total_n_pos
        if total_n_pos > 0
        else float("nan"),
        "val_loss_neg": total_loss_neg / total_n_neg
        if total_n_neg > 0
        else float("nan"),
        "val_n_pos": total_n_pos,
        "val_n_neg": total_n_neg,
        "val_conditioning_accuracy": total_correct_wins / total_samples,
        "val_conditioning_gap": total_gap / total_samples,
        "val_conditioning_gap_pos": total_gap_pos / total_n_pos
        if total_n_pos > 0
        else float("nan"),
        "val_conditioning_gap_neg": total_gap_neg / total_n_neg
        if total_n_neg > 0
        else float("nan"),
        "val_adv_episode_alignment": total_aligned / total_samples,
        "val_alignment_on_success": (
            total_aligned_success / total_success_samples
            if total_success_samples > 0
            else float("nan")
        ),
        "val_alignment_on_failure": (
            total_aligned_failure / total_failure_samples
            if total_failure_samples > 0
            else float("nan")
        ),
    }


def _log_val_metrics(tag: str, metrics: dict[str, float]) -> None:
    logging.info(
        f"[{tag}] "
        f"val_loss={metrics['val_loss']:.5f} "
        f"(pos={metrics['val_loss_pos']:.5f}, neg={metrics['val_loss_neg']:.5f}) "
        f"cond_acc={metrics['val_conditioning_accuracy']:.4f} "
        f"cond_gap={metrics['val_conditioning_gap']:.5f} "
        f"(gap_pos={metrics['val_conditioning_gap_pos']:.5f}, "
        f"gap_neg={metrics['val_conditioning_gap_neg']:.5f}) "
        f"adv_ep_align={metrics['val_adv_episode_alignment']:.4f} "
        f"(success={metrics['val_alignment_on_success']:.4f}, "
        f"failure={metrics['val_alignment_on_failure']:.4f}) "
        f"n_pos={metrics['val_n_pos']} n_neg={metrics['val_n_neg']}"
    )


# ── Checkpoint save ──────────────────────────────────────────────────────────


def save_pistar_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: RECAPPiStarTrainingConfig,
    policy: PiStar06Policy,
    optimizer,
    scheduler,
    preprocessor=None,
    postprocessor=None,
    metrics: dict | None = None,
) -> None:
    """Write a resumable checkpoint in lerobot's pretrained/training_state layout."""

    pretrained_dir = checkpoint_dir / "pretrained_model"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(pretrained_dir)
    write_json(draccus.encode(cfg), pretrained_dir / "train_config.json")
    if metrics is not None:
        write_json(metrics, pretrained_dir / "metrics.json")
    if preprocessor is not None:
        preprocessor.save_pretrained(pretrained_dir)
    if postprocessor is not None:
        postprocessor.save_pretrained(pretrained_dir)
    save_training_state(checkpoint_dir, step, optimizer, scheduler)


# ── Main training loop ───────────────────────────────────────────────────────


@parser.wrap()
def run_recap_pistar_train_val(cfg: RECAPPiStarTrainingConfig) -> None:
    """Train/validate PiStar06 with RECAP conditioning and stratified metrics."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    set_seed(cfg.seed)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
        mixed_precision="bf16",
    )
    is_main = accelerator.is_main_process
    device = accelerator.device

    # Compute output_dir on main, then broadcast so all ranks share the same path.
    # Only main writes to it; typing it as a real Path everywhere avoids
    # `Path | None` operator issues in main-only branches.
    if is_main:
        output_dir = Path("outputs/pistar") / datetime.now().strftime(
            "%Y-%m-%d/%H-%M-%S"
        )
    else:
        output_dir = Path()
    output_dir_buf: list[Path] = [output_dir]
    broadcast_object_list(output_dir_buf, from_process=0)
    output_dir = output_dir_buf[0]
    checkpoints_dir = output_dir / "checkpoints"

    if is_main:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        write_json(draccus.encode(cfg), output_dir / "train_config.json")
        logging.info(f"Using device: {device} (world_size={accelerator.num_processes})")

    wandb_run = _init_wandb(cfg) if is_main else None

    # ── 1. Load dataset and build episode-level train/val split ──────────
    if is_main:
        full_dataset = LeRobotDataset(repo_id=cfg.dataset_repo_id, vcodec="auto")
    accelerator.wait_for_everyone()
    if not is_main:
        full_dataset = LeRobotDataset(repo_id=cfg.dataset_repo_id, vcodec="auto")

    success_by_episode = load_episode_success_from_dataset(full_dataset)
    logging.info(
        f"Loaded success labels for {len(success_by_episode)} episodes "
        "from the dataset's 'success' column."
    )

    vn_meta = None
    if cfg.policy.enable_advantage_conditioning:
        # Pull num_value_bins, c_fail, and reward config from the value network
        # so the return targets used here match what the VN was trained against.
        vn_meta = load_vn_metadata(cfg.advantage.value_network_pretrained_path)
        logging.info(
            f"Pulled from value network: c_fail={vn_meta.c_fail} "
            f"num_value_bins={vn_meta.num_value_bins} reward={vn_meta.reward}"
        )
    else:
        logging.info(
            "Advantage conditioning DISABLED — training vanilla Pi0.5 "
            "(no value network, no advantage text injection)"
        )
        cfg.policy.advantage_dropout = 1.0

    step_rewards: dict[int, float] | None = None
    if vn_meta is not None and vn_meta.reward is not None:
        if is_main:
            logging.info(
                "Reconstructing per-step rewards from VN reward config "
                f"(type={vn_meta.reward.type})"
            )
            step_rewards = vn_meta.reward.compute_step_rewards(full_dataset, device)
        accelerator.wait_for_everyone()
        if not is_main:
            step_rewards = vn_meta.reward.compute_step_rewards(full_dataset, device)

    c_fail = vn_meta.c_fail if vn_meta is not None else 50.0
    num_value_bins = vn_meta.num_value_bins if vn_meta is not None else 50
    frame_targets = build_frame_targets(
        dataset=full_dataset,
        success_by_episode=success_by_episode,
        c_fail=c_fail,
        num_value_bins=num_value_bins,
        step_rewards=step_rewards,
    )
    train_targets, val_targets = split_train_val_targets(
        frame_targets=frame_targets,
        val_ratio=cfg.val_split_ratio,
        seed=cfg.seed,
    )

    train_ep_ids = sorted({t.episode_index for t in train_targets})
    val_ep_ids = sorted({t.episode_index for t in val_targets})
    logging.info(
        f"Split: {len(train_ep_ids)} train episodes ({len(train_targets)} frames), "
        f"{len(val_ep_ids)} val episodes ({len(val_targets)} frames)"
    )

    # ── 2. Build policy config and preprocessor ────────────────────────────
    policy_cfg = _resolve_policy_config(cfg, full_dataset, c_fail=c_fail)
    if is_main:
        _log_memory("post-dataset-split")

    # ── 3. Pre-compute advantages using Pi0.5-based value network ────────
    if cfg.policy.enable_advantage_conditioning:
        assert vn_meta is not None
        advantage_lookup, new_threshold = prepare_advantages(
            cfg=cfg.advantage,
            dataset=full_dataset,
            policy_cfg=policy_cfg,
            vn_meta=vn_meta,
            success_by_episode=success_by_episode,
            frame_targets=frame_targets,
            accelerator=accelerator,
            num_workers=cfg.num_workers,
        )
        if new_threshold is not None:
            cfg.policy.advantage_threshold = new_threshold
        else:
            logging.info(
                f"Using fixed advantage_threshold={cfg.policy.advantage_threshold:.5f} "
                f"(advantage.threshold_percentile is None)"
            )
        if is_main:
            _log_memory("post-advantage-precompute")
    else:
        advantage_lookup: dict[int, float] = {}
        logging.info(
            "Skipping advantage pre-computation (advantage conditioning disabled)"
        )

    # ── 4. Create separate datasets for train and val ────────────────────
    delta_timestamps = resolve_delta_timestamps(policy_cfg, full_dataset.meta)

    del full_dataset, frame_targets, train_targets, val_targets
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_dataset = LeRobotDataset(
        repo_id=cfg.dataset_repo_id,
        episodes=train_ep_ids,
        delta_timestamps=delta_timestamps,
        vcodec="auto",
    )
    val_dataset = LeRobotDataset(
        repo_id=cfg.dataset_repo_id,
        episodes=val_ep_ids,
        delta_timestamps=delta_timestamps,
        vcodec="auto",
    )

    # ── 5. Create policy ─────────────────────────────────────────────────
    # Initialize model weights in the target precision to halve peak memory
    # and speed up weight init (random normal on smaller tensors).
    target_dtype = torch.bfloat16 if cfg.policy.dtype == "bfloat16" else torch.float32
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(target_dtype)
    try:
        policy = PiStar06Policy(config=policy_cfg)
    finally:
        torch.set_default_dtype(original_dtype)
    policy.recap_log_every = cfg.log_every_n_steps
    if is_main:
        _log_memory("post-policy-init")

    if cfg.policy.pretrained_path is not None:
        pretrained_path = str(cfg.policy.pretrained_path)
        logging.info(f"Loading pretrained Pi0.5 weights from {pretrained_path}")
        from safetensors.torch import load_file

        local_safetensors = Path(pretrained_path).expanduser() / "model.safetensors"
        if local_safetensors.is_file():
            resolved_file = str(local_safetensors)
        else:
            from transformers.utils import cached_file

            resolved_file = cached_file(pretrained_path, "model.safetensors")
        raw_sd = load_file(resolved_file)  # ty:ignore[invalid-argument-type]
        fixed_sd = policy._fix_pytorch_state_dict_keys(raw_sd, policy.config)
        del raw_sd

        remapped_sd = {
            (f"model.{k}" if not k.startswith("model.") else k): v
            for k, v in fixed_sd.items()
        }
        del fixed_sd

        missing, unexpected = policy.load_state_dict(remapped_sd, strict=False)
        del remapped_sd
        gc.collect()

        if missing:
            logging.info(
                f"Missing keys when loading pretrained: {len(missing)} "
                "(expected for advantage_embedding)"
            )
        if unexpected:
            logging.warning(
                f"Unexpected keys when loading pretrained: {len(unexpected)}"
            )
    if is_main:
        _log_memory("post-pretrained-load")

    policy.to(device)
    if is_main:
        _log_memory("post-policy-to-device")

    num_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in policy.parameters())
    logging.info(f"Trainable parameters: {num_trainable:,} / {num_total:,} total")

    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

    preprocessor, postprocessor = make_pi05_pre_post_processors(
        config=policy_cfg,
        dataset_stats=train_dataset.meta.stats,  # ty: ignore[invalid-argument-type]
    )

    env_preprocessor = None
    env_postprocessor = None
    if cfg.sim_eval_every_n_train_steps > 0:
        is_libero_plus = isinstance(cfg.eval_cfg, LiberoPlusEvalConfig)
        rep_env_cfg = LiberoEnv(
            task=cfg.eval_cfg.suites[0],
            fps=cfg.eval_cfg.fps,
            observation_height=cfg.eval_cfg.observation_height,
            observation_width=cfg.eval_cfg.observation_width,
            is_libero_plus=is_libero_plus,
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=rep_env_cfg, policy_cfg=policy_cfg
        )
        mode = "LIBERO-plus" if is_libero_plus else "LIBERO"
        logging.info(f"{mode} sim eval every {cfg.sim_eval_every_n_train_steps} steps")

    # ── 6. Create dataloaders ────────────────────────────────────────────
    train_sampler = None
    train_shuffle = True
    drop_n_last_frames = getattr(policy_cfg, "drop_n_last_frames", None)
    if drop_n_last_frames is not None:
        train_shuffle = False
        train_sampler = EpisodeAwareSampler(
            train_dataset.meta.episodes["dataset_from_index"],
            train_dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=train_dataset.episodes,
            drop_n_last_frames=int(drop_n_last_frames),
            shuffle=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # ── 7. Optimizer and scheduler (pi05 presets) ───────────────────────
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    trainable_params = policy.get_optim_params()
    pi05_defaults = PI05Config()
    optimizer_preset = pi05_defaults.get_optimizer_preset()
    scheduler_preset = pi05_defaults.get_scheduler_preset()
    max_grad_norm = optimizer_preset.grad_clip_norm
    optimizer = optimizer_preset.build(trainable_params)
    scheduler = scheduler_preset.build(optimizer, num_training_steps=cfg.train_steps)
    if is_main:
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

    # ── 7.5. Wrap policy/optimizer/loader/scheduler for distributed ──────
    # val_loader is intentionally not prepared — validation runs on main rank only.
    policy, optimizer, train_loader, scheduler = accelerator.prepare(
        policy, optimizer, train_loader, scheduler
    )

    # ── 8. Training loop ─────────────────────────────────────────────────
    best_pc_success = -1.0
    history: list[dict] = []
    skipped_batches = 0
    nan_val_metrics = {
        "val_loss": float("nan"),
        "val_loss_pos": float("nan"),
        "val_loss_neg": float("nan"),
        "val_n_pos": 0,
        "val_n_neg": 0,
        "val_conditioning_accuracy": float("nan"),
        "val_conditioning_gap": float("nan"),
        "val_conditioning_gap_pos": float("nan"),
        "val_conditioning_gap_neg": float("nan"),
        "val_adv_episode_alignment": float("nan"),
        "val_alignment_on_success": float("nan"),
        "val_alignment_on_failure": float("nan"),
    }

    if is_main:
        logging.info(
            f"Starting training: {cfg.train_steps} steps, "
            f"{len(train_dataset)} train frames, {len(val_dataset)} val frames"
        )
        _log_memory("pre-training-loop")

    train_iter = cycle(train_loader)
    policy.train()
    optimizer.zero_grad(set_to_none=True)

    start_time = time_module.perf_counter()
    last_log_step = 0
    last_log_time = start_time

    for global_step in range(1, cfg.train_steps + 1):
        # Pull a batch with retry on known video decode errors.
        while True:
            try:
                batch = next(train_iter)
                break
            except RuntimeError as exc:
                if not is_known_video_validation_error(exc):
                    raise
                skipped_batches += 1
                logging.warning(
                    f"[step {global_step}] Skipping training batch due to "
                    f"video decode error (skipped {skipped_batches} so far): {exc}"
                )

        batch = preprocessor(batch)
        batch = inject_advantages(batch, advantage_lookup, device)

        with accelerator.autocast():
            loss, output_dict = policy.forward(batch)

        accelerator.backward(loss)
        if max_grad_norm > 0:
            accelerator.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if global_step == 1 and is_main:
            _log_memory("first-train-step")

        step_loss = float(loss.item())
        wandb_step_metrics: dict[str, float] = {}

        # ── Sim eval (independent cadence, main rank only) ────────────────
        is_sim_eval_step = (
            cfg.sim_eval_every_n_train_steps > 0
            and global_step % cfg.sim_eval_every_n_train_steps == 0
        )
        if is_sim_eval_step and is_main:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if isinstance(cfg.eval_cfg, LiberoPlusEvalConfig):
                step_eval_metrics = run_libero_plus_eval(
                    cfg.eval_cfg,
                    policy=unwrapped_policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    seed=cfg.seed,
                    videos_dir=output_dir / "eval" / f"videos_step_{global_step}",
                    wandb_run=wandb_run,
                    wandb_step=global_step,
                )
            else:
                step_eval_metrics = run_libero_eval(
                    cfg.eval_cfg,
                    policy=unwrapped_policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    seed=cfg.seed,
                    videos_dir=output_dir / "eval" / f"videos_step_{global_step}",
                    wandb_run=wandb_run,
                    wandb_step=global_step,
                )
            wandb_step_metrics.update(
                {f"eval/{k}": v for k, v in step_eval_metrics.items()}
            )

            pc_success = step_eval_metrics["pc_success"]
            if pc_success > best_pc_success:
                best_pc_success = pc_success
                save_pistar_checkpoint(
                    checkpoint_dir=checkpoints_dir / "best",
                    step=global_step,
                    cfg=cfg,
                    policy=unwrapped_policy,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    metrics=step_eval_metrics,
                )
                logging.info(f"New best pc_success: {best_pc_success:.4f}")

            policy.train()
        if is_sim_eval_step:
            accelerator.wait_for_everyone()

        is_log_step = (
            global_step == 1
            or global_step % cfg.log_every_n_steps == 0
            or global_step == cfg.train_steps
        )
        if not is_log_step:
            if wandb_run is not None and wandb_step_metrics:
                wandb_run.log(wandb_step_metrics, step=global_step)
            continue

        # ── Train log ────────────────────────────────────────────────
        now = time_module.perf_counter()
        elapsed = now - start_time
        window_elapsed = max(now - last_log_time, 1e-9)
        steps_per_sec = (global_step - last_log_step) / window_elapsed
        eta = max(cfg.train_steps - global_step, 0) / max(steps_per_sec, 1e-9)
        lr = optimizer.param_groups[0]["lr"]

        logging.info(
            f"[step {global_step}/{cfg.train_steps}] "
            f"loss={step_loss:.5f} "
            f"lr={lr:.2e} "
            f"it/s={steps_per_sec:.2f} "
            f"elapsed={format_duration(elapsed)} "
            f"eta={format_duration(eta)}"
        )
        wandb_step_metrics.update(
            {
                "train/loss": step_loss,
                "train/lr": lr,
                "train/steps_per_sec": steps_per_sec,
                "global_step": global_step,
            }
        )

        # ── Validate (on sim-eval cadence to amortise its ~25% wall-time cost) ──
        is_val_step = cfg.sim_eval_every_n_train_steps > 0 and (
            global_step % cfg.sim_eval_every_n_train_steps == 0
            or global_step == cfg.train_steps
        )
        if is_val_step and is_main:
            try:
                val_metrics = _run_validation(
                    accelerator.unwrap_model(policy),
                    val_loader,
                    preprocessor,
                    advantage_lookup,
                    success_by_episode,
                    device,
                    max_steps=cfg.max_val_steps,
                )
            except Exception as error:  # noqa: BLE001
                if not is_known_video_validation_error(error):
                    policy.train()
                    raise
                logging.warning(
                    f"[step {global_step}] Validation skipped due to persistent "
                    f"video decoding/timestamp errors: {error}"
                )
                val_metrics = dict(nan_val_metrics)
            else:
                _log_val_metrics(f"step {global_step}/{cfg.train_steps}", val_metrics)
                wandb_step_metrics.update(
                    {f"val/{k}": v for k, v in val_metrics.items()}
                )
            policy.train()
        else:
            val_metrics = dict(nan_val_metrics)
        if is_val_step:
            accelerator.wait_for_everyone()

        # ── History + checkpoint (main rank only) ────────────────────
        saved_metrics = {
            "global_step": global_step,
            "train_loss": step_loss,
            "lr": lr,
            **val_metrics,
        }
        if is_main:
            history.append(saved_metrics)
            write_json(history, output_dir / "metrics_history.json")  # ty: ignore[invalid-argument-type]

            save_pistar_checkpoint(
                checkpoint_dir=checkpoints_dir / "last",
                step=global_step,
                cfg=cfg,
                policy=accelerator.unwrap_model(policy),
                optimizer=optimizer,
                scheduler=scheduler,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                metrics=saved_metrics,
            )

        if wandb_run is not None and wandb_step_metrics:
            wandb_run.log(wandb_step_metrics, step=global_step)

        # Reset window AFTER validation/checkpoint so their wall time is not
        # counted as training throughput in the next window.
        last_log_step = global_step
        last_log_time = time_module.perf_counter()

    if skipped_batches > 0:
        logging.warning(
            f"Skipped {skipped_batches} training batches due to video decode errors"
        )
    accelerator.wait_for_everyone()

    if is_main:
        logging.info(f"Training complete. Best pc_success: {best_pc_success:.4f}")
        logging.info(
            f"Final checkpoint at {checkpoints_dir / 'last' / 'pretrained_model'}"
        )

        if cfg.push_to_hub:
            from huggingface_hub import HfApi

            repo_id = f"reece-omahoney/{cfg.job_name}"
            logging.info(f"Pushing PiStar06 policy to hub: {repo_id}")
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_to_hub(repo_id)
            preprocessor.push_to_hub(repo_id)
            postprocessor.push_to_hub(repo_id)
            HfApi().upload_file(
                path_or_fileobj=str(
                    checkpoints_dir / "last" / "pretrained_model" / "train_config.json"
                ),
                path_in_repo="train_config.json",
                repo_id=repo_id,
                repo_type="model",
            )

        if wandb_run is not None:
            wandb_run.finish()

    accelerator.end_training()


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    run_recap_pistar_train_val()  # ty: ignore[missing-argument]
