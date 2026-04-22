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
import json
import logging
import resource
import time as time_module
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from lerobot.configs import parser
from lerobot.configs.default import EvalConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.envs.configs import EnvConfig, LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS
from lerobot_policy_pistar06.modeling_pistar06 import PiStar06Policy
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from distal import advantage_cache
from distal import train_value as base
from distal.value_model import RECAPValueConfig, RECAPValueNetwork


def _log_memory(label: str) -> None:
    """Log CPU RSS and CUDA memory at a named checkpoint."""
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    parts = [f"[MEM {label}] CPU_RSS={rss_mb:.0f}MB"]
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        parts.append(f"CUDA_alloc={alloc:.0f}MB CUDA_reserved={reserved:.0f}MB")
    logging.info(" ".join(parts))


@dataclass
class RECAPPiStarTrainingConfig:
    """Configuration for RECAP PiStar06 advantage-conditioned Pi0.5 policy training."""

    repo_id: str = "reece-omahoney/pi05-libero-10"
    value_network_pretrained_path: str = "reece-omahoney/value-maha-pi05-paligemma"
    root: str | None = None
    revision: str | None = None
    episodes: list[int] | None = None

    epochs: int = 5
    batch_size: int = 64
    num_workers: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    val_split_ratio: float = 0.1
    seed: int = 42
    device: str = "auto"
    max_train_steps_per_epoch: int | None = None
    max_val_steps_per_epoch: int | None = None
    log_every_n_steps: int = 10
    validate_every_n_train_steps: int = 1000
    max_val_steps_per_step_validation: int | None = 50
    sim_eval_every_n_train_steps: int = 500

    # Master switch: set to False to train vanilla Pi0.5 without advantage
    # text injection (baseline for ablation experiments).
    enable_advantage_conditioning: bool = True

    # RECAP advantage conditioning
    c_fail: float = 50.0
    num_value_bins: int = 50
    # Per-frame advantage threshold: only frames with advantage > threshold get
    # "Advantage: positive" text.  The paper (Appendix A.4) sets this to a
    # per-task percentile so that ~30% of frames are positive during pre-training
    # and ~40% during fine-tuning.  Use advantage_threshold_percentile to compute
    # this automatically from the advantage distribution; when set it overrides
    # advantage_threshold after advantages are pre-computed.
    advantage_threshold: float = 0.0
    advantage_threshold_percentile: float | None = 70.0
    advantage_dropout: float = 0.3
    cfg_beta: float = 1.0

    # Pi0.5 model settings
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    num_expert_layers: int = 0
    pretrained_path: str = "lerobot/pi05-libero"
    model_precision: str = "bfloat16"
    freeze_vision_encoder: bool = True
    freeze_backbone: bool = True
    num_unfrozen_backbone_layers: int = 3
    train_expert_only: bool = False
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 1

    # Value network pre-computation
    vn_batch_size: int = 640

    # Advantage caching: content-addressed cache keyed by a hash of the
    # inputs that determine the precomputed advantages. Cache files are
    # mirrored on HF Hub under ``advantage_cache_repo_id`` (set to None to
    # disable remote cache). Bump ``advantage_cache_schema_version`` to
    # invalidate all existing caches after a pipeline change that file
    # hashes can't detect.
    advantage_cache_repo_id: str | None = "reece-omahoney/advantage-caches"
    advantage_cache_local_dir: str = "outputs"
    advantage_cache_schema_version: int = 1

    # Sim eval (defaults mirror configs/eval.yaml)
    env: EnvConfig | None = field(
        default_factory=lambda: LiberoEnv(
            task="libero_10", fps=20, observation_height=256, observation_width=256
        )
    )
    eval: EvalConfig = field(
        default_factory=lambda: EvalConfig(n_episodes=20, batch_size=10)
    )
    use_async_envs: bool = True
    max_parallel_tasks: int = 1

    # Hub push for trained PiStar06 policy
    pi_star_repo_id: str | None = "reece-omahoney/pistar06-libero-maha"
    push_to_hub: bool = True

    # Weights & Biases (optional; set wandb_project to enable)
    wandb_project: str | None = "distal"
    wandb_entity: str | None = None
    wandb_run_name: str | None = "pistar06-libero-maha"


def _init_wandb(cfg: RECAPPiStarTrainingConfig):
    """Initialise a W&B run if ``wandb_project`` is set, otherwise return ``None``."""
    if cfg.wandb_project is None:
        return None
    import wandb

    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_run_name,
        config=asdict(cfg),
    )
    logging.info(f"W&B run: {run.url}")
    return run


def _load_vn_train_config(path_or_repo_id: str) -> dict:
    """Best-effort load of train_config.json from a local dir or HF repo."""
    local = Path(path_or_repo_id).expanduser() / "train_config.json"
    if local.is_file():
        config_path = str(local)
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            config_path = hf_hub_download(
                repo_id=path_or_repo_id,
                filename="train_config.json",
                repo_type="model",
            )
        except (EntryNotFoundError, FileNotFoundError):
            logging.warning(
                f"No train_config.json in '{path_or_repo_id}'; "
                "cannot auto-resolve c_fail."
            )
            return {}
    with open(config_path) as f:
        return json.load(f)


def _build_policy_config(
    cfg: RECAPPiStarTrainingConfig,
    train_dataset: LeRobotDataset,
):
    """Build a PiStar06Config from the training config and dataset metadata."""
    from lerobot_policy_pistar06.configuration_pistar06 import PiStar06Config

    features = dataset_to_policy_features(train_dataset.meta.features)
    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }

    policy_cfg = PiStar06Config(
        input_features=input_features,
        output_features=output_features,
        paligemma_variant=cfg.paligemma_variant,
        action_expert_variant=cfg.action_expert_variant,
        num_expert_layers=cfg.num_expert_layers,
        dtype=cfg.model_precision,
        freeze_vision_encoder=cfg.freeze_vision_encoder,
        train_expert_only=cfg.train_expert_only,
        gradient_checkpointing=cfg.gradient_checkpointing,
        c_fail=cfg.c_fail,
        advantage_threshold=cfg.advantage_threshold,
        advantage_dropout=cfg.advantage_dropout,
        cfg_beta=cfg.cfg_beta,
        enable_advantage_conditioning=cfg.enable_advantage_conditioning,
    )
    return policy_cfg


# ── Backbone freezing ────────────────────────────────────────────────────────


def _apply_backbone_freezing(policy, cfg: RECAPPiStarTrainingConfig) -> None:
    """Freeze the PaliGemma VLM backbone, optionally unfreezing the last N layers.

    Mirrors the partial-unfreeze pattern used in RECAPValueNetwork.
    """
    if not cfg.freeze_backbone:
        return

    paligemma = policy.model.paligemma_with_expert.paligemma
    paligemma.eval()
    for param in paligemma.parameters():
        param.requires_grad = False

    if cfg.num_unfrozen_backbone_layers > 0:
        lm = paligemma.model.language_model
        lm_inner = lm.model if hasattr(lm, "model") else lm
        layers = lm_inner.layers
        num_layers = len(layers)
        if cfg.num_unfrozen_backbone_layers > num_layers:
            raise ValueError(
                f"num_unfrozen_backbone_layers={cfg.num_unfrozen_backbone_layers} "
                f"exceeds available layers {num_layers}"
            )
        unfrozen = layers[-cfg.num_unfrozen_backbone_layers :]
        for layer in unfrozen:
            layer.train()
            for param in layer.parameters():
                param.requires_grad = True
        logging.info(
            "Backbone frozen; unfreezing last "
            f"{cfg.num_unfrozen_backbone_layers}/{num_layers} "
            "VLM language model layers"
        )
    else:
        logging.info("Backbone fully frozen (all PaliGemma params)")


def _restore_freeze_state(policy, cfg: RECAPPiStarTrainingConfig) -> None:
    """Re-apply eval() to frozen backbone parts after a policy.train() call."""
    if not cfg.freeze_backbone:
        return

    paligemma = policy.model.paligemma_with_expert.paligemma
    paligemma.eval()

    if cfg.num_unfrozen_backbone_layers > 0:
        lm = paligemma.model.language_model
        lm_inner = lm.model if hasattr(lm, "model") else lm
        for layer in lm_inner.layers[-cfg.num_unfrozen_backbone_layers :]:
            layer.train()


# ── Advantage pre-computation ────────────────────────────────────────────────


def _make_vn_preprocessor(policy_cfg, dataset_stats):
    """Build a lightweight preprocessor for VN advantage precomputation.

    Omits DeviceProcessorStep (VN handles device transfer internally) and
    AddBatchDimensionProcessorStep (DataLoader already batches).
    """
    from lerobot.policies.pi05.processor_pi05 import (
        Pi05PrepareStateTokenizerProcessorStep,
    )
    from lerobot.processor import (
        NormalizerProcessorStep,
        PolicyProcessorPipeline,
        TokenizerProcessorStep,
    )

    steps = [
        NormalizerProcessorStep(
            features={**policy_cfg.input_features, **policy_cfg.output_features},
            norm_map=policy_cfg.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=policy_cfg.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=policy_cfg.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
    ]
    return PolicyProcessorPipeline(steps=steps, name="vn_preprocessor")


@torch.no_grad()
def _precompute_advantages(
    full_dataset: LeRobotDataset,
    frame_targets: list[base.FrameTarget],
    value_network: RECAPValueNetwork,
    policy_cfg,
    device: torch.device,
    batch_size: int = 4,
) -> tuple[dict[int, float], dict[int, int]]:
    """Pre-compute per-frame advantages using the frozen value network.

    Builds a lightweight preprocessor internally that produces only the
    fields consumed by ``RECAPValueNetwork`` (language tokens + images),
    without moving every tensor to GPU.
    """
    preprocessor = _make_vn_preprocessor(policy_cfg, full_dataset.meta.stats)

    vn = value_network
    vn.eval()
    vn.to(device)
    for param in vn.parameters():
        param.requires_grad = False
    logging.info(
        f"Value network loaded: {sum(p.numel() for p in vn.parameters()):,} params"
    )
    _log_memory("post-VN-load")

    R_t_by_abs_index: dict[int, float] = {}
    for ft in frame_targets:
        abs_idx = int(full_dataset.hf_dataset[ft.frame_index]["index"])
        R_t_by_abs_index[abs_idx] = ft.target_value

    loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    advantage_lookup: dict[int, float] = {}
    episode_lookup: dict[int, int] = {}
    total_frames = 0

    for batch in loader:
        abs_indices = batch["index"]
        ep_indices = batch["episode_index"]
        B = abs_indices.shape[0]

        batch = preprocessor(batch)
        outputs = vn.compute_outputs(batch)
        V_t = outputs["expected_value"].squeeze(-1).cpu()

        for i in range(B):
            abs_idx = int(abs_indices[i].item())
            R_t = R_t_by_abs_index.get(abs_idx)
            if R_t is not None:
                advantage_lookup[abs_idx] = R_t - V_t[i].item()
                episode_lookup[abs_idx] = int(ep_indices[i].item())

        total_frames += B
        if total_frames % 500 == 0:
            logging.info(
                f"  Pre-computed advantages for {total_frames}/"
                f"{len(full_dataset)} frames"
            )

    logging.info(
        f"Advantage pre-computation complete: {len(advantage_lookup)} frames, "
        f"mean={sum(advantage_lookup.values()) / max(1, len(advantage_lookup)):.4f}"
    )

    del vn, preprocessor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return advantage_lookup, episode_lookup


def _compute_advantage_threshold(
    advantage_lookup: dict[int, float],
    percentile: float,
) -> float:
    """Compute an advantage threshold from the Nth percentile of the distribution.

    Following the paper (Appendix A.4): during pre-training the threshold is set
    so that ~30% of frames are positive (percentile=70); during fine-tuning ~40%
    are positive (percentile=60).
    """
    import numpy as np

    values = np.array(list(advantage_lookup.values()))
    threshold = float(np.percentile(values, percentile))
    pct_positive = float((values > threshold).sum()) / len(values) * 100

    logging.info(
        f"Advantage distribution ({len(values)} frames): "
        f"mean={values.mean():.5f} std={values.std():.5f} "
        f"min={values.min():.5f} max={values.max():.5f}"
    )
    logging.info(
        f"Auto-threshold from {percentile:.0f}th percentile: {threshold:.5f} "
        f"({pct_positive:.1f}% of frames will be positive)"
    )
    return threshold


def _inject_advantages(
    batch: dict,
    advantage_lookup: dict[int, float],
    device: torch.device,
) -> dict:
    """Inject pre-computed advantages into a batch dict."""
    abs_indices = batch["index"]
    advantages = torch.tensor(
        [advantage_lookup.get(int(idx.item()), 0.0) for idx in abs_indices],
        dtype=torch.float32,
        device=device,
    )
    batch["advantage"] = advantages
    return batch


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
            if not base._is_known_video_validation_error(exc):
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
        batch = _inject_advantages(batch, advantage_lookup, device)

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


def _run_sim_eval(
    *,
    policy,
    eval_env,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    cfg: RECAPPiStarTrainingConfig,
    step: int,
    output_dir: Path,
    wandb_run=None,
) -> dict[str, float]:
    """Roll out the policy in sim, log videos to W&B, return overall metrics."""
    logging.info(f"Running sim eval at global_step={step}")
    policy.eval()
    with torch.no_grad():
        eval_info = eval_policy_all(
            envs=eval_env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            videos_dir=output_dir / "eval" / f"videos_step_{step}",
            max_episodes_rendered=4,
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.max_parallel_tasks,
        )
    aggregated = eval_info["overall"]
    for suite, suite_info in eval_info.items():
        logging.info(f"Suite {suite} aggregated: {suite_info}")

    if wandb_run is not None:
        import wandb

        fps = getattr(cfg.env, "fps", 30) if cfg.env is not None else 30
        video_paths = aggregated.get("video_paths") or []
        if video_paths:
            wandb_run.log(
                {"eval/video": wandb.Video(str(video_paths[0]), fps=fps, format="mp4")},
                step=step,
            )

    return {
        "eval_s": aggregated.get("eval_s", 0.0),
        "avg_sum_reward": aggregated.get("avg_sum_reward", 0.0),
        "pc_success": aggregated.get("pc_success", 0.0),
    }


# ── Main training loop ───────────────────────────────────────────────────────


@parser.wrap()
def run_recap_pistar_train_val(cfg: RECAPPiStarTrainingConfig) -> None:
    """Train/validate PiStar06 with RECAP conditioning and stratified metrics."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    base._set_seed(cfg.seed)

    output_dir = Path("outputs/pistar") / datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    base._save_json(output_dir / "train_config.json", asdict(cfg))

    device = base._resolve_device(cfg.device)
    logging.info(f"Using device: {device}")

    wandb_run = _init_wandb(cfg)

    # ── 1. Load dataset and build episode-level train/val split ──────────
    full_dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
        root=cfg.root,
        revision=cfg.revision,
        episodes=cfg.episodes,
    )

    success_by_episode = base._load_episode_success_from_dataset(full_dataset)
    logging.info(
        f"Loaded success labels for {len(success_by_episode)} episodes "
        "from the dataset's 'success' column."
    )

    if cfg.enable_advantage_conditioning:
        # Pull num_value_bins from the VN policy config and c_fail from its
        # train_config.json so the return targets used here match what the
        # value network was trained against.
        vn_policy_cfg = RECAPValueConfig.from_pretrained(
            cfg.value_network_pretrained_path
        )
        if vn_policy_cfg.num_value_bins != cfg.num_value_bins:
            logging.warning(
                f"Overriding num_value_bins from {cfg.num_value_bins} -> "
                f"{vn_policy_cfg.num_value_bins} to match value network"
            )
            cfg.num_value_bins = int(vn_policy_cfg.num_value_bins)

        vn_train_cfg = _load_vn_train_config(cfg.value_network_pretrained_path)
        vn_c_fail = vn_train_cfg.get("c_fail")
        if vn_c_fail is not None and vn_c_fail != cfg.c_fail:
            logging.warning(
                f"Overriding c_fail from {cfg.c_fail} -> {vn_c_fail} "
                "to match value network"
            )
            cfg.c_fail = float(vn_c_fail)
    else:
        logging.info(
            "Advantage conditioning DISABLED — training vanilla Pi0.5 "
            "(no value network, no advantage text injection)"
        )
        cfg.advantage_dropout = 1.0

    frame_targets = base._build_frame_targets(
        dataset=full_dataset,
        success_by_episode=success_by_episode,
        c_fail=cfg.c_fail,
        num_value_bins=cfg.num_value_bins,
    )
    train_targets, val_targets = base._split_train_val_targets(
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
    policy_cfg = _build_policy_config(cfg, full_dataset)
    policy_cfg.n_action_steps = 10
    _log_memory("post-dataset-split")

    # ── 3. Pre-compute advantages using Pi0.5-based value network ────────
    if cfg.enable_advantage_conditioning:
        signature = advantage_cache.compute_signature(
            schema_version=cfg.advantage_cache_schema_version,
            dataset_repo_id=cfg.repo_id,
            dataset_revision=cfg.revision,
            episodes=cfg.episodes,
            value_network_pretrained_path=cfg.value_network_pretrained_path,
            c_fail=cfg.c_fail,
            num_value_bins=cfg.num_value_bins,
        )
        logging.info(f"Advantage cache signature: {signature}")

        cache_file: Path | None = None
        if cfg.advantage_cache_repo_id:
            cache_file = advantage_cache.try_download(
                cfg.advantage_cache_repo_id, signature
            )

        if cache_file is not None:
            advantage_lookup, _ = advantage_cache.load(cache_file)
        else:
            vn_model = RECAPValueNetwork.from_pretrained(
                cfg.value_network_pretrained_path
            )
            advantage_lookup, episode_lookup = _precompute_advantages(
                full_dataset=full_dataset,
                frame_targets=frame_targets,
                value_network=vn_model,
                policy_cfg=policy_cfg,
                device=device,
                batch_size=cfg.vn_batch_size,
            )

            local_cache_path = (
                Path(cfg.advantage_cache_local_dir)
                / f"advantage_cache_{signature}.json"
            )
            advantage_cache.save(
                local_cache_path,
                advantage_lookup,
                episode_lookup=episode_lookup,
                metadata={
                    "signature": signature,
                    "schema_version": cfg.advantage_cache_schema_version,
                    "value_network_pretrained_path": cfg.value_network_pretrained_path,
                    "c_fail": cfg.c_fail,
                    "num_value_bins": cfg.num_value_bins,
                    "repo_id": cfg.repo_id,
                    "success_by_episode": success_by_episode,
                },
            )
            if cfg.advantage_cache_repo_id:
                advantage_cache.upload(
                    local_cache_path, cfg.advantage_cache_repo_id, signature
                )
        _log_memory("post-advantage-precompute")

        # ── 3b. Auto-compute advantage threshold from percentile ─────────
        if cfg.advantage_threshold_percentile is not None:
            cfg.advantage_threshold = _compute_advantage_threshold(
                advantage_lookup, cfg.advantage_threshold_percentile
            )
            policy_cfg.advantage_threshold = cfg.advantage_threshold
        else:
            logging.info(
                f"Using fixed advantage_threshold={cfg.advantage_threshold:.5f} "
                f"(advantage_threshold_percentile is None)"
            )
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
        repo_id=cfg.repo_id,
        root=cfg.root,
        revision=cfg.revision,
        episodes=train_ep_ids,
        delta_timestamps=delta_timestamps,
    )
    val_dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
        root=cfg.root,
        revision=cfg.revision,
        episodes=val_ep_ids,
        delta_timestamps=delta_timestamps,
    )

    # ── 5. Create policy ─────────────────────────────────────────────────
    # Initialize model weights in the target precision to halve peak memory
    # and speed up weight init (random normal on smaller tensors).
    target_dtype = (
        torch.bfloat16 if cfg.model_precision == "bfloat16" else torch.float32
    )
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(target_dtype)
    try:
        policy = PiStar06Policy(config=policy_cfg)
    finally:
        torch.set_default_dtype(original_dtype)
    _log_memory("post-policy-init")

    if cfg.pretrained_path is not None:
        logging.info(f"Loading pretrained Pi0.5 weights from {cfg.pretrained_path}")
        from safetensors.torch import load_file

        local_safetensors = Path(cfg.pretrained_path).expanduser() / "model.safetensors"
        if local_safetensors.is_file():
            resolved_file = str(local_safetensors)
        else:
            from transformers.utils import cached_file

            resolved_file = cached_file(cfg.pretrained_path, "model.safetensors")
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
    _log_memory("post-pretrained-load")

    policy.to(device)
    _apply_backbone_freezing(policy, cfg)
    _log_memory("post-policy-to-device")

    num_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in policy.parameters())
    logging.info(f"Trainable parameters: {num_trainable:,} / {num_total:,} total")

    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

    preprocessor, postprocessor = make_pi05_pre_post_processors(
        config=policy_cfg,
        dataset_stats=train_dataset.meta.stats,  # ty: ignore[invalid-argument-type]
    )

    eval_env = None
    env_preprocessor = None
    env_postprocessor = None
    if cfg.env is not None:
        logging.info("Creating sim eval env")
        eval_env = make_env(
            cfg.env,
            n_envs=cfg.eval.batch_size,
            use_async_envs=cfg.use_async_envs,
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=cfg.env,
            policy_cfg=policy_cfg,
        )

    # ── 6. Create dataloaders ────────────────────────────────────────────
    train_sampler = None
    train_shuffle = True
    if hasattr(policy_cfg, "drop_n_last_frames"):
        train_shuffle = False
        train_sampler = EpisodeAwareSampler(
            train_dataset.meta.episodes["dataset_from_index"],
            train_dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=train_dataset.episodes,
            drop_n_last_frames=policy_cfg.drop_n_last_frames,
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
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    step_val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # ── 7. Optimizer and scheduler ──────────────────────────────────────
    trainable_params = policy.get_optim_params()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg.epochs),
    )

    # ── 8. Training loop ─────────────────────────────────────────────────
    best_val_cond_acc = -1.0
    history: list[dict] = []
    global_train_step = 0

    if cfg.validate_every_n_train_steps < 0:
        raise ValueError(
            "validate_every_n_train_steps must be >= 0, got "
            f"{cfg.validate_every_n_train_steps}"
        )

    logging.info(
        f"Starting training: {cfg.epochs} epochs, "
        f"{len(train_dataset)} train frames, {len(val_dataset)} val frames"
    )
    _log_memory("pre-training-loop")

    for epoch in range(1, cfg.epochs + 1):
        policy.train()
        _restore_freeze_state(policy, cfg)
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_start = time_module.perf_counter()
        skipped_batches = 0

        train_iter = iter(train_loader)
        step = -1
        while True:
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            except RuntimeError as exc:
                if not base._is_known_video_validation_error(exc):
                    raise
                skipped_batches += 1
                logging.warning(
                    f"[Epoch {epoch}] Skipping training batch due to video "
                    f"decode error (skipped {skipped_batches} so far): {exc}"
                )
                continue
            step += 1

            if (
                cfg.max_train_steps_per_epoch is not None
                and step >= cfg.max_train_steps_per_epoch
            ):
                break

            batch = preprocessor(batch)
            batch = _inject_advantages(batch, advantage_lookup, device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss, output_dict = policy.forward(batch)
                if cfg.gradient_accumulation_steps > 1:
                    loss = loss / cfg.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0 or step == len(
                train_loader
            ) - 1:
                if cfg.max_grad_norm > 0:
                    clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            unscaled_loss = loss.item() * (
                cfg.gradient_accumulation_steps
                if cfg.gradient_accumulation_steps > 1
                else 1
            )
            epoch_loss += unscaled_loss * batch[ACTION].shape[0]
            epoch_samples += batch[ACTION].shape[0]
            global_train_step += 1

            if global_train_step == 1:
                _log_memory("first-train-step")

            wandb_step_metrics: dict[str, float] = {}

            if (
                cfg.log_every_n_steps > 0
                and global_train_step % cfg.log_every_n_steps == 0
            ):
                avg_loss = (
                    epoch_loss / epoch_samples if epoch_samples > 0 else float("nan")
                )
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time_module.perf_counter() - epoch_start
                logging.info(
                    f"[Epoch {epoch}/{cfg.epochs} step {step + 1}] "
                    f"train_loss={avg_loss:.5f} lr={lr:.2e} elapsed={elapsed:.1f}s "
                    f"global_step={global_train_step}"
                )
                wandb_step_metrics.update(
                    {
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/step_loss": loss.item(),
                        "global_step": global_train_step,
                    }
                )

            # Step-based validation
            if (
                cfg.validate_every_n_train_steps > 0
                and global_train_step % cfg.validate_every_n_train_steps == 0
            ):
                step_val_max = (
                    cfg.max_val_steps_per_step_validation
                    if cfg.max_val_steps_per_step_validation is not None
                    else cfg.max_val_steps_per_epoch
                )
                step_val_metrics = _run_validation(
                    policy,
                    step_val_loader,
                    preprocessor,
                    advantage_lookup,
                    success_by_episode,
                    device,
                    max_steps=step_val_max,
                )
                tag = (
                    f"Epoch {epoch}/{cfg.epochs} step-validate "
                    f"(global_step={global_train_step})"
                )
                _log_val_metrics(tag, step_val_metrics)
                wandb_step_metrics.update(
                    {f"val/{k}": v for k, v in step_val_metrics.items()}
                )
                policy.train()
                _restore_freeze_state(policy, cfg)

            # Step-based sim eval
            if (
                eval_env is not None
                and cfg.sim_eval_every_n_train_steps > 0
                and global_train_step % cfg.sim_eval_every_n_train_steps == 0
            ):
                step_eval_metrics = _run_sim_eval(
                    policy=policy,
                    eval_env=eval_env,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    cfg=cfg,
                    step=global_train_step,
                    output_dir=output_dir,
                    wandb_run=wandb_run,
                )
                wandb_step_metrics.update(
                    {f"eval/{k}": v for k, v in step_eval_metrics.items()}
                )
                policy.train()
                _restore_freeze_state(policy, cfg)

            if wandb_run is not None and wandb_step_metrics:
                wandb_run.log(wandb_step_metrics, step=global_train_step)

        train_loss = epoch_loss / epoch_samples if epoch_samples > 0 else float("nan")

        if skipped_batches > 0:
            logging.warning(
                f"[Epoch {epoch}] Skipped {skipped_batches} training batches "
                "due to video decode errors"
            )

        # End-of-epoch validation
        try:
            val_metrics = _run_validation(
                policy,
                val_loader,
                preprocessor,
                advantage_lookup,
                success_by_episode,
                device,
                max_steps=cfg.max_val_steps_per_epoch,
            )
        except Exception as error:  # noqa: BLE001
            if not base._is_known_video_validation_error(error):
                raise
            logging.warning(
                f"[Epoch {epoch}] Validation failed with video error; "
                "retrying with num_workers=0."
            )
            try:
                val_metrics = _run_validation(
                    policy,
                    step_val_loader,
                    preprocessor,
                    advantage_lookup,
                    success_by_episode,
                    device,
                    max_steps=cfg.max_val_steps_per_epoch,
                )
            except Exception as retry_error:  # noqa: BLE001
                if not base._is_known_video_validation_error(retry_error):
                    raise
                logging.warning(f"[Epoch {epoch}] Validation skipped: {retry_error}")
                val_metrics = {
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

        scheduler.step()

        _log_val_metrics(f"Epoch {epoch}/{cfg.epochs} epoch-end", val_metrics)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **val_metrics,
        }
        history.append(epoch_metrics)

        logging.info(
            f"[Epoch {epoch}/{cfg.epochs}] "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_metrics['val_loss']:.5f} "
            f"cond_acc={val_metrics['val_conditioning_accuracy']:.4f} "
            f"cond_gap={val_metrics['val_conditioning_gap']:.5f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {f"epoch/{k}": v for k, v in epoch_metrics.items()},
                step=global_train_step,
            )

        base._save_json(output_dir / "metrics_history.json", history)

        checkpoint = {
            "epoch": epoch,
            "global_train_step": global_train_step,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "policy_config": policy_cfg,
            "train_config": asdict(cfg),
            "metrics": epoch_metrics,
        }
        torch.save(checkpoint, checkpoints_dir / "last.pt")

        cond_acc = val_metrics["val_conditioning_accuracy"]
        if not (cond_acc != cond_acc) and cond_acc > best_val_cond_acc:  # noqa: PLR0124 (NaN check)
            best_val_cond_acc = cond_acc
            torch.save(checkpoint, checkpoints_dir / "best.pt")
            logging.info(f"New best conditioning accuracy: {best_val_cond_acc:.4f}")

    logging.info(
        f"Training complete. Best val conditioning accuracy: {best_val_cond_acc:.4f}"
    )

    if eval_env is not None:
        close_envs(eval_env)

    # ── 9. Export in HuggingFace pretrained format for inference ──────────
    pretrained_dir = output_dir / "pretrained"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(pretrained_dir)
    logging.info(f"Saved pretrained model to {pretrained_dir}")

    if cfg.push_to_hub and cfg.pi_star_repo_id:
        logging.info(f"Pushing PiStar06 policy to hub: {cfg.pi_star_repo_id}")
        policy.push_to_hub(cfg.pi_star_repo_id)
        preprocessor.push_to_hub(cfg.pi_star_repo_id)
        postprocessor.push_to_hub(cfg.pi_star_repo_id)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    run_recap_pistar_train_val()  # ty: ignore[missing-argument]
