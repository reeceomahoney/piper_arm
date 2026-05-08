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

import numpy as np
import torch
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.utils.io_utils import write_json
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import cycle
from lerobot_policy_pistar06.configuration_pistar06 import PiStar06Config
from lerobot_policy_pistar06.modeling_pistar06 import PiStar06Policy
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from distal import advantage_cache
from distal.sim_eval import resolve_eval_task_ids, run_sim_eval
from distal.train_value import (
    FrameTarget,
    build_frame_targets,
    format_duration,
    is_known_video_validation_error,
    load_episode_success_from_dataset,
    split_train_val_targets,
)
from distal.value_model import RECAPValueConfig, RECAPValueNetwork


@dataclass
class AdvantageConfig:
    """Runtime advantage-conditioning options.

    Policy-level knobs (enable_advantage_conditioning, c_fail,
    advantage_threshold, advantage_dropout, cfg_beta) live on PiStar06Config
    and are accessed via ``cfg.policy.*`` — override on the CLI as e.g.
    ``--policy.advantage_threshold=0.05``.

    ``c_fail`` and ``num_value_bins`` are pulled from the value network at
    runtime and are not user-facing knobs.
    """

    value_network_pretrained_path: str = "reece-omahoney/value-maha-libero-plus"

    # When set, override ``policy.advantage_threshold`` with the Nth percentile
    # of the precomputed advantage distribution (~30% positive at p=70).
    threshold_percentile: float | None = 70.0

    # Value network advantage precomputation batch size
    vn_batch_size: int = 640

    # Cache pre-computed advantages locally at
    # ``$HF_ASSETS_CACHE/distal/advantages/<signature>.json``, content-addressed
    # by every input that affects the result. Set to False to always recompute.
    cache: bool = True


@dataclass
class RECAPPiStarTrainingConfig:
    """Configuration for RECAP PiStar06 advantage-conditioned Pi0.5 policy training."""

    job_name: str = "pistar06-libero-plus-maha"
    repo_id: str = "reece-omahoney/pi05-libero-plus"

    train_steps: int = 20_000
    batch_size: int = 64
    num_workers: int = 8
    val_split_ratio: float = 0.1
    seed: int = 42
    device: str = "auto"
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

    # Sim eval — fat AsyncVectorEnv eval over LIBERO suites.  Two modes:
    #
    # - is_libero_plus=True (default): mirrors ``distal/collect_libero_plus.py``.
    #   Each chunk packs up to ``eval_parallel_envs`` distinct task IDs (1 env
    #   each) into one vec env. ``eval_per_cell`` and ``eval_task_seed`` MUST
    #   equal the values used at collect time so eval rolls out the same task
    #   IDs that appear in the rollout dataset (defaults match collect's
    #   per_cell=1, seed=0).
    # - is_libero_plus=False: standard LIBERO eval. Each suite has 10 tasks;
    #   each task gets its own vec env with ``eval_n_envs_per_task`` sub-envs
    #   (distinct ``episode_index`` → distinct init states).
    #
    # Set sim_eval_every_n_train_steps=0 to disable sim eval entirely.
    is_libero_plus: bool = True
    eval_suites: list[str] = field(default_factory=lambda: ["libero_goal"])
    eval_fps: int = 20
    eval_observation_height: int = 256
    eval_observation_width: int = 256
    eval_per_cell: int = 1
    eval_task_seed: int = 0
    eval_max_tasks: int | None = None
    # libero-plus only: restrict eval to a single base task (after stripping
    # perturbation suffixes), e.g. "turn_on_the_stove". When set, sampled task
    # IDs are filtered to only those whose base name matches.
    eval_base_task: str | None = "turn_on_the_stove"
    eval_parallel_envs: int = 0  # libero-plus only; 0 = auto-scale by CPU cores
    eval_n_envs_per_task: int = 0  # base-LIBERO only
    eval_n_episodes_per_task: int = 1

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


# ── Advantage pre-computation ────────────────────────────────────────────────


def _make_vn_preprocessor(policy_cfg, dataset_stats, tokenizer_name: str):
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
            tokenizer_name=tokenizer_name,
            max_length=policy_cfg.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
    ]
    return PolicyProcessorPipeline(steps=steps, name="vn_preprocessor")


@torch.no_grad()
def _precompute_advantages(
    full_dataset: LeRobotDataset,
    frame_targets: list[FrameTarget],
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
    preprocessor = _make_vn_preprocessor(
        policy_cfg, full_dataset.meta.stats, value_network.config.text_backbone
    )

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
        model_batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        V_t = vn.predict_value(model_batch).cpu()  # ty: ignore[missing-argument, invalid-argument-type]

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

    output_dir = Path("outputs/pistar") / datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    write_json(asdict(cfg), output_dir / "train_config.json")

    device = get_safe_torch_device(cfg.device, log=True)
    logging.info(f"Using device: {device}")

    wandb_run = _init_wandb(cfg)

    # ── 1. Load dataset and build episode-level train/val split ──────────
    full_dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
        vcodec="auto",
    )

    success_by_episode = load_episode_success_from_dataset(full_dataset)
    logging.info(
        f"Loaded success labels for {len(success_by_episode)} episodes "
        "from the dataset's 'success' column."
    )

    vn_reward_mode: str = "steps"
    vn_maha_stats_path: str | None = None
    vn_base_policy: str | None = None
    c_fail: float = 50.0
    num_value_bins: int = 50
    if cfg.policy.enable_advantage_conditioning:
        # Pull num_value_bins, c_fail, and reward config from the value network
        # so the return targets used here match what the VN was trained against.
        vn_policy_cfg = PreTrainedConfig.from_pretrained(
            cfg.advantage.value_network_pretrained_path
        )
        assert isinstance(vn_policy_cfg, RECAPValueConfig)
        num_value_bins = int(vn_policy_cfg.num_value_bins)

        vn_train_cfg = _load_vn_train_config(
            cfg.advantage.value_network_pretrained_path
        )
        c_fail = float(vn_train_cfg.get("c_fail", c_fail))

        vn_reward_cfg = vn_train_cfg.get("reward") or {}
        vn_reward_mode = str(vn_reward_cfg.get("type", "steps"))
        vn_maha_stats_path = vn_reward_cfg.get("stats_path")
        vn_base_policy = vn_reward_cfg.get("base_policy")
        logging.info(
            f"Pulled from value network: c_fail={c_fail} "
            f"num_value_bins={num_value_bins} "
            f"reward.type={vn_reward_mode!r} "
            f"stats_path={vn_maha_stats_path!r} "
            f"base_policy={vn_base_policy!r}"
        )
        if vn_reward_mode == "maha" and vn_maha_stats_path is None:
            raise ValueError(
                "Value network was trained with reward.type='maha' but its "
                "train_config.json is missing reward.stats_path; cannot "
                "reconstruct the per-step rewards used to build R_t."
            )
    else:
        logging.info(
            "Advantage conditioning DISABLED — training vanilla Pi0.5 "
            "(no value network, no advantage text injection)"
        )
        cfg.policy.advantage_dropout = 1.0

    step_rewards: dict[int, float] | None = None
    if cfg.policy.enable_advantage_conditioning and vn_reward_mode == "maha":
        from distal.rewards.maha import load_or_compute_maha_rewards

        embed_policy_path = vn_base_policy or str(cfg.policy.pretrained_path)
        logging.info(
            "Loading or computing Mahalanobis-distance rewards to match the "
            f"value network's training signal (embed policy: {embed_policy_path}, "
            f"stats: {vn_maha_stats_path})"
        )
        step_rewards = load_or_compute_maha_rewards(
            dataset=full_dataset,
            policy_path=embed_policy_path,
            stats_path=vn_maha_stats_path,  # ty: ignore[invalid-argument-type]
            device=device,
            batch_size=int(vn_reward_cfg.get("embed_batch_size", 32)),
            num_workers=int(vn_reward_cfg.get("embed_num_workers", 4)),
        )

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
    _log_memory("post-dataset-split")

    # ── 3. Pre-compute advantages using Pi0.5-based value network ────────
    if cfg.policy.enable_advantage_conditioning:
        signature = advantage_cache.compute_signature(
            dataset_repo_id=cfg.repo_id,
            value_network_pretrained_path=cfg.advantage.value_network_pretrained_path,
            c_fail=c_fail,
            num_value_bins=num_value_bins,
            reward_mode=vn_reward_mode,
            maha_stats_path=vn_maha_stats_path,
        )
        cache_file = advantage_cache.cache_path(signature)
        logging.info(f"Advantage cache signature: {signature} -> {cache_file}")

        if cfg.advantage.cache and cache_file.is_file():
            advantage_lookup, _ = advantage_cache.load(cache_file)
        else:
            vn_model = RECAPValueNetwork.from_pretrained(
                cfg.advantage.value_network_pretrained_path
            )
            advantage_lookup, episode_lookup = _precompute_advantages(
                full_dataset=full_dataset,
                frame_targets=frame_targets,
                value_network=vn_model,
                policy_cfg=policy_cfg,
                device=device,
                batch_size=cfg.advantage.vn_batch_size,
            )
            if cfg.advantage.cache:
                advantage_cache.save(
                    cache_file,
                    advantage_lookup,
                    episode_lookup=episode_lookup,
                    metadata={
                        "signature": signature,
                        "value_network_pretrained_path": (
                            cfg.advantage.value_network_pretrained_path
                        ),
                        "c_fail": c_fail,
                        "num_value_bins": num_value_bins,
                        "reward_mode": vn_reward_mode,
                        "maha_stats_path": vn_maha_stats_path,
                        "repo_id": cfg.repo_id,
                        "success_by_episode": success_by_episode,
                    },
                )
        _log_memory("post-advantage-precompute")

        # ── 3b. Auto-compute advantage threshold from percentile ─────────
        if cfg.advantage.threshold_percentile is not None:
            cfg.policy.advantage_threshold = _compute_advantage_threshold(
                advantage_lookup, cfg.advantage.threshold_percentile
            )
        else:
            logging.info(
                f"Using fixed advantage_threshold={cfg.policy.advantage_threshold:.5f} "
                f"(advantage.threshold_percentile is None)"
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
        episodes=train_ep_ids,
        delta_timestamps=delta_timestamps,
        vcodec="auto",
    )
    val_dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
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
    _log_memory("post-pretrained-load")

    policy.to(device)
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
        rep_env_cfg = LiberoEnv(
            task=cfg.eval_suites[0],
            fps=cfg.eval_fps,
            observation_height=cfg.eval_observation_height,
            observation_width=cfg.eval_observation_width,
            is_libero_plus=cfg.is_libero_plus,
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=rep_env_cfg, policy_cfg=policy_cfg
        )
        if cfg.is_libero_plus:
            logging.info(
                f"LIBERO-plus sim eval every {cfg.sim_eval_every_n_train_steps} "
                f"steps (suites={cfg.eval_suites}, per_cell={cfg.eval_per_cell}, "
                f"task_seed={cfg.eval_task_seed}, "
                f"base_task={cfg.eval_base_task})"
            )
            for suite in cfg.eval_suites:
                ids = resolve_eval_task_ids(
                    suite,
                    per_cell=cfg.eval_per_cell,
                    task_seed=cfg.eval_task_seed,
                    base_task=cfg.eval_base_task,
                    max_tasks=cfg.eval_max_tasks,
                )
                logging.info(f"  {suite}: {len(ids)} task IDs")
        else:
            logging.info(
                f"Base LIBERO sim eval every {cfg.sim_eval_every_n_train_steps} "
                f"steps (suites={cfg.eval_suites}, "
                f"n_envs_per_task={cfg.eval_n_envs_per_task}, "
                f"n_ep_per_task={cfg.eval_n_episodes_per_task})"
            )

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
        batch = _inject_advantages(batch, advantage_lookup, device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            loss, output_dict = policy.forward(batch)

        loss.backward()
        if max_grad_norm > 0:
            clip_grad_norm_(trainable_params, max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if global_step == 1:
            _log_memory("first-train-step")

        step_loss = float(loss.item())
        wandb_step_metrics: dict[str, float] = {}

        # ── Sim eval (independent cadence) ────────────────────────────
        if (
            cfg.sim_eval_every_n_train_steps > 0
            and global_step % cfg.sim_eval_every_n_train_steps == 0
        ):
            step_eval_metrics = run_sim_eval(
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                suites=cfg.eval_suites,
                is_libero_plus=cfg.is_libero_plus,
                fps=cfg.eval_fps,
                observation_height=cfg.eval_observation_height,
                observation_width=cfg.eval_observation_width,
                per_cell=cfg.eval_per_cell,
                task_seed=cfg.eval_task_seed,
                base_task=cfg.eval_base_task,
                max_tasks=cfg.eval_max_tasks,
                parallel_envs=cfg.eval_parallel_envs,
                n_envs_per_task=cfg.eval_n_envs_per_task,
                n_episodes_per_task=cfg.eval_n_episodes_per_task,
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
                best_checkpoint = {
                    "global_train_step": global_step,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "policy_config": policy_cfg,
                    "train_config": asdict(cfg),
                    "metrics": step_eval_metrics,
                }
                torch.save(best_checkpoint, checkpoints_dir / "best.pt")
                logging.info(f"New best pc_success: {best_pc_success:.4f}")

            policy.train()

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

        # ── Validate ─────────────────────────────────────────────────
        try:
            val_metrics = _run_validation(
                policy,
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
            wandb_step_metrics.update({f"val/{k}": v for k, v in val_metrics.items()})

        policy.train()

        # ── History + checkpoint ─────────────────────────────────────
        saved_metrics = {
            "global_step": global_step,
            "train_loss": step_loss,
            "lr": lr,
            **val_metrics,
        }
        history.append(saved_metrics)
        write_json(history, output_dir / "metrics_history.json")  # ty: ignore[invalid-argument-type]

        checkpoint = {
            "global_train_step": global_step,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "policy_config": policy_cfg,
            "train_config": asdict(cfg),
            "metrics": saved_metrics,
        }
        torch.save(checkpoint, checkpoints_dir / "last.pt")

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
    logging.info(f"Training complete. Best pc_success: {best_pc_success:.4f}")

    # ── 9. Export in HuggingFace pretrained format for inference ──────────
    pretrained_dir = output_dir / "pretrained"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(pretrained_dir)
    logging.info(f"Saved pretrained model to {pretrained_dir}")

    if cfg.push_to_hub:
        repo_id = f"reece-omahoney/{cfg.job_name}"
        logging.info(f"Pushing PiStar06 policy to hub: {repo_id}")
        policy.push_to_hub(repo_id)
        preprocessor.push_to_hub(repo_id)
        postprocessor.push_to_hub(repo_id)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    run_recap_pistar_train_val()  # ty: ignore[missing-argument]
