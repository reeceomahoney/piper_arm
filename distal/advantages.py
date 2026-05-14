"""Per-frame advantage labels for PiStar06 training.

Everything about turning a frozen value network into a per-frame advantage
lookup lives here: VN metadata resolution, preprocessor for VN inference,
the precompute loop, the content-addressed cache, optional per-task threshold
shifting, and batch injection during training/validation.

The high-level entry point is :func:`prepare_advantages`. ``train_pi_star``
calls it once before the training loop; ``compare_task_thresholds`` uses the
lower-level helpers directly.
"""

import gc
import json
import logging
import time as time_module
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_policy_pistar06.configuration_pistar06 import PiStar06Config
from torch.utils.data import DataLoader

from distal import advantage_cache
from distal.collect_libero_plus import base_task_name
from distal.rewards.configs import MahaRewardConfig, RewardConfig
from distal.train_value import (
    FrameTarget,
    build_episode_infos,
    format_duration,
)
from distal.value_model import RECAPValueConfig, RECAPValueNetwork
from distal.variant_names import try_load_variant_names

# ── Config ───────────────────────────────────────────────────────────────────


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

    value_network_pretrained_path: str = "reece-omahoney/value-knn-rel-libero-plus"

    # When set, override ``policy.advantage_threshold`` with the Nth percentile
    # of the precomputed advantage distribution (~30% positive at p=70).
    threshold_percentile: float | None = 70.0

    # When True, compute the percentile threshold per base task and shift each
    # frame's advantage by its task threshold. For LIBERO-plus datasets,
    # ``meta/variant_names.json`` (written by collect_libero_plus.py, or
    # backfilled by distal/backfill_variant_names.py) is used to collapse
    # variants to base tasks; otherwise grouping falls back to the dataset's
    # raw task string.
    per_task_threshold: bool = True

    # Value network advantage precomputation batch size
    vn_batch_size: int = 640

    # Cache pre-computed advantages locally at
    # ``$HF_ASSETS_CACHE/distal/advantages/<signature>.json``, content-addressed
    # by every input that affects the result. Set to False to always recompute.
    cache: bool = True


@dataclass(frozen=True)
class ValueNetworkMetadata:
    """Frozen metadata read off the VN at runtime (not user-facing knobs).

    These values determine both the return targets used to train the VN and
    the cache signature for advantages computed against it, so they must
    match between the VN repo and downstream consumers.
    """

    c_fail: float
    num_value_bins: int
    reward: RewardConfig | None

    @property
    def reward_mode(self) -> str:
        return self.reward.type if self.reward is not None else "steps"

    @property
    def maha_stats_path(self) -> str | None:
        return (
            self.reward.stats_path
            if isinstance(self.reward, MahaRewardConfig)
            else None
        )


def load_vn_train_config(path_or_repo_id: str) -> dict:
    """Best-effort load of ``train_config.json`` from a local dir or HF repo."""
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


def load_vn_metadata(vn_pretrained_path: str) -> ValueNetworkMetadata:
    """Resolve c_fail, num_value_bins, and reward config from a VN checkpoint."""
    vn_policy_cfg = PreTrainedConfig.from_pretrained(vn_pretrained_path)
    assert isinstance(vn_policy_cfg, RECAPValueConfig)
    num_value_bins = int(vn_policy_cfg.num_value_bins)

    train_cfg = load_vn_train_config(vn_pretrained_path)
    c_fail = float(train_cfg.get("c_fail", 50.0))

    reward: RewardConfig | None = None
    reward_dict = train_cfg.get("reward")
    if reward_dict:
        reward = draccus.decode(RewardConfig, reward_dict)

    return ValueNetworkMetadata(
        c_fail=c_fail, num_value_bins=num_value_bins, reward=reward
    )


# ── VN preprocessor + precompute ─────────────────────────────────────────────


def make_vn_preprocessor(policy_cfg, dataset_stats, tokenizer_name: str):
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
def precompute_advantages(
    full_dataset: LeRobotDataset,
    frame_targets: list[FrameTarget],
    value_network: RECAPValueNetwork,
    policy_cfg,
    device: torch.device,
    batch_size: int = 4,
    num_workers: int = 0,
) -> tuple[dict[int, float], dict[int, int]]:
    """Pre-compute per-frame advantages using the frozen value network.

    Builds a lightweight preprocessor internally that produces only the
    fields consumed by ``RECAPValueNetwork`` (language tokens + images),
    without moving every tensor to GPU.
    """
    preprocessor = make_vn_preprocessor(
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

    R_t_by_abs_index: dict[int, float] = {}
    for ft in frame_targets:
        abs_idx = int(full_dataset.hf_dataset[ft.frame_index]["index"])
        R_t_by_abs_index[abs_idx] = ft.target_value

    loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    advantage_lookup: dict[int, float] = {}
    episode_lookup: dict[int, int] = {}
    total_frames = 0
    n_total = len(full_dataset)
    log_every_n_batches = 10
    start_time = time_module.perf_counter()

    for batch_idx, batch in enumerate(loader):
        abs_indices = batch["index"]
        ep_indices = batch["episode_index"]
        B = abs_indices.shape[0]

        batch = preprocessor(batch)
        model_batch = {
            k: v.to(device, non_blocking=True)
            for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }
        V_t = vn.predict_value(model_batch).cpu()  # ty: ignore[missing-argument, invalid-argument-type]

        for i in range(B):
            abs_idx = int(abs_indices[i].item())
            R_t = R_t_by_abs_index.get(abs_idx)
            if R_t is not None:
                advantage_lookup[abs_idx] = R_t - V_t[i].item()
                episode_lookup[abs_idx] = int(ep_indices[i].item())

        total_frames += B
        if (batch_idx + 1) % log_every_n_batches == 0 or total_frames >= n_total:
            elapsed = time_module.perf_counter() - start_time
            frames_per_sec = total_frames / max(elapsed, 1e-9)
            eta = max(n_total - total_frames, 0) / max(frames_per_sec, 1e-9)
            logging.info(
                f"  Pre-computed advantages for {total_frames}/{n_total} frames "
                f"({frames_per_sec:.1f} frames/s, "
                f"elapsed={format_duration(elapsed)}, "
                f"eta={format_duration(eta)})"
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


# ── Threshold computation ────────────────────────────────────────────────────


def compute_advantage_threshold(
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


def build_abs_to_task(dataset: LeRobotDataset) -> dict[int, str]:
    """Map every absolute frame index to a stable per-task label.

    For LIBERO-plus datasets (those with a ``meta/variant_names.json``
    sidecar) we collapse variants to base tasks via ``base_task_name``,
    because the dataset's stored ``task`` string is the rewritten natural
    language for language perturbations and so doesn't identify the
    underlying task. For base LIBERO the dataset's ``task`` string already
    names the base task and is used directly.
    """
    ep_to_variant = try_load_variant_names(dataset)
    ep_to_task: dict[int, str] | None = None
    if ep_to_variant is not None:
        ep_to_task = {ep: base_task_name(v) for ep, v in ep_to_variant.items()}
        logging.info(
            f"Loaded variant_names sidecar ({len(ep_to_variant)} episodes); "
            f"grouping by base_task_name → {len({*ep_to_task.values()})} tasks"
        )
    else:
        logging.info(
            "No variant_names sidecar found — grouping by the dataset's raw task string"
        )

    episode_infos = build_episode_infos(dataset)
    abs_to_task: dict[int, str] = {}
    for info in episode_infos.values():
        task = ep_to_task[info.episode_index] if ep_to_task is not None else info.task
        for abs_idx in range(info.start_index, info.end_index):
            abs_to_task[abs_idx] = task
    return abs_to_task


def compute_per_task_thresholds(
    advantage_lookup: dict[int, float],
    abs_to_task: dict[int, str],
    percentile: float,
) -> dict[str, float]:
    """Compute the Nth percentile of advantages within each task group."""
    by_task: dict[str, list[float]] = {}
    for abs_idx, adv in advantage_lookup.items():
        task = abs_to_task.get(abs_idx)
        if task is None:
            continue
        by_task.setdefault(task, []).append(adv)

    thresholds = {
        task: float(np.percentile(np.asarray(vals, dtype=np.float64), percentile))
        for task, vals in by_task.items()
    }

    all_values = np.asarray(list(advantage_lookup.values()), dtype=np.float64)
    global_th = float(np.percentile(all_values, percentile))
    th_values = np.asarray(list(thresholds.values()), dtype=np.float64)
    logging.info(
        f"Per-task thresholds @ p{percentile:.0f} across {len(thresholds)} tasks: "
        f"global_th={global_th:+.5f}  "
        f"per_task range [{th_values.min():+.5f}, {th_values.max():+.5f}]  "
        f"mean_diff={(th_values - global_th).mean():+.5f}  "
        f"std_diff={(th_values - global_th).std():.5f}"
    )
    for task in sorted(thresholds):
        n = len(by_task[task])
        diff = thresholds[task] - global_th
        logging.info(
            f"  {task}: n={n} th={thresholds[task]:+.5f} diff_vs_global={diff:+.5f}"
        )
    return thresholds


def shift_advantages_by_task(
    advantage_lookup: dict[int, float],
    abs_to_task: dict[int, str],
    task_thresholds: dict[str, float],
) -> dict[int, float]:
    """Subtract each frame's task threshold from its advantage.

    After this shift, ``adv > 0`` equals ``raw_adv > task_threshold`` per
    frame, so the policy's existing ``advantage > advantage_threshold`` check
    works with ``advantage_threshold=0``. Frames whose task is missing from
    the thresholds dict are left unshifted (defensive fallback).
    """
    shifted: dict[int, float] = {}
    missing = 0
    for abs_idx, adv in advantage_lookup.items():
        task = abs_to_task.get(abs_idx)
        if task is None or task not in task_thresholds:
            missing += 1
            shifted[abs_idx] = adv
            continue
        shifted[abs_idx] = adv - task_thresholds[task]
    if missing:
        logging.warning(
            f"shift_advantages_by_task: {missing} frames had no task threshold "
            "and were left unshifted"
        )
    return shifted


# ── Batch injection ──────────────────────────────────────────────────────────


def inject_advantages(
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


# ── High-level entry point ───────────────────────────────────────────────────


def prepare_advantages(
    cfg: AdvantageConfig,
    dataset: LeRobotDataset,
    policy_cfg: PiStar06Config,
    vn_meta: ValueNetworkMetadata,
    success_by_episode: dict[int, int],
    frame_targets: list[FrameTarget],
    accelerator: Accelerator,
    num_workers: int = 0,
) -> tuple[dict[int, float], float | None]:
    """End-to-end advantage preparation for ``train_pi_star``.

    Steps:

    1. Compute cache signature from dataset + VN metadata.
    2. If cached, load on every rank (cheap JSON parse). Otherwise rank 0 runs
       the VN forward pass and broadcasts the result to the other ranks.
    3. If ``threshold_percentile`` is set: optionally shift each advantage by
       its base-task percentile threshold (then the scalar threshold the
       policy should use is ``0.0``); otherwise compute a single global
       threshold over the whole distribution.

    Returns ``(advantage_lookup, scalar_threshold)``. ``scalar_threshold`` is
    ``None`` when ``threshold_percentile`` is unset — the caller should keep
    whatever value lives on ``policy.advantage_threshold`` in that case.
    """
    is_main = accelerator.is_main_process
    device = accelerator.device

    signature = advantage_cache.compute_signature(
        dataset_repo_id=dataset.repo_id,
        value_network_pretrained_path=cfg.value_network_pretrained_path,
        c_fail=vn_meta.c_fail,
        num_value_bins=vn_meta.num_value_bins,
        reward_mode=vn_meta.reward_mode,
        maha_stats_path=vn_meta.maha_stats_path,
    )
    cache_file = advantage_cache.cache_path(signature)
    logging.info(f"Advantage cache signature: {signature} -> {cache_file}")

    advantage_lookup: dict[int, float] | None
    if cfg.cache and cache_file.is_file():
        advantage_lookup, _ = advantage_cache.load(cache_file)
    else:
        if is_main:
            vn_model = RECAPValueNetwork.from_pretrained(
                cfg.value_network_pretrained_path
            )
            advantage_lookup, episode_lookup = precompute_advantages(
                full_dataset=dataset,
                frame_targets=frame_targets,
                value_network=vn_model,
                policy_cfg=policy_cfg,
                device=device,
                batch_size=cfg.vn_batch_size,
                num_workers=num_workers,
            )
            if cfg.cache:
                advantage_cache.save(
                    cache_file,
                    advantage_lookup,
                    episode_lookup=episode_lookup,
                    metadata={
                        "signature": signature,
                        "value_network_pretrained_path": (
                            cfg.value_network_pretrained_path
                        ),
                        "c_fail": vn_meta.c_fail,
                        "num_value_bins": vn_meta.num_value_bins,
                        "reward_mode": vn_meta.reward_mode,
                        "maha_stats_path": vn_meta.maha_stats_path,
                        "dataset_repo_id": dataset.repo_id,
                        "success_by_episode": success_by_episode,
                    },
                )
        else:
            advantage_lookup = None
        buf = [advantage_lookup]
        broadcast_object_list(buf, from_process=0)
        advantage_lookup = buf[0]
        assert advantage_lookup is not None, "broadcast advantage_lookup is None"

    threshold: float | None = None
    if cfg.threshold_percentile is not None:
        if cfg.per_task_threshold:
            abs_to_task = build_abs_to_task(dataset)
            task_thresholds = compute_per_task_thresholds(
                advantage_lookup, abs_to_task, cfg.threshold_percentile
            )
            advantage_lookup = shift_advantages_by_task(
                advantage_lookup, abs_to_task, task_thresholds
            )
            threshold = 0.0
        else:
            threshold = compute_advantage_threshold(
                advantage_lookup, cfg.threshold_percentile
            )
    return advantage_lookup, threshold


# Re-export ``field`` so callers don't need ``dataclasses.field`` just to
# customise default mutable AdvantageConfig fields.
__all__ = [
    "AdvantageConfig",
    "ValueNetworkMetadata",
    "build_abs_to_task",
    "compute_advantage_threshold",
    "compute_per_task_thresholds",
    "field",
    "inject_advantages",
    "load_vn_metadata",
    "load_vn_train_config",
    "make_vn_preprocessor",
    "precompute_advantages",
    "prepare_advantages",
    "shift_advantages_by_task",
]
