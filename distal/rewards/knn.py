"""kNN-distance-based per-step rewards for value training.

Mirror of ``distal/rewards/maha.py`` but with the per-frame distance computed
as the mean distance from each rollout VLM embedding to its k nearest
neighbours in a base-policy demo dataset, instead of a fitted Gaussian's
Mahalanobis distance.

Demo embeddings are cached locally (content-addressed by policy + dataset +
subsample) under ``demo_embs_cache_dir`` so reruns skip the expensive vision
pass. Per-frame normalised rewards are also cached locally (content-addressed
by everything that affects them) via ``rewards.maha.load_or_compute_rewards``.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.processor import rename_stats
from safetensors.numpy import load_file, save_file
from torch.utils.data import DataLoader, Subset

from distal.rewards.maha_stats import embed_siglip_pooled


def embed_dataset(
    policy: PI05Policy,
    preprocessor,
    dataset: LeRobotDataset | Subset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_frames: int | None,
    subsample_seed: int,
    desc: str,
) -> np.ndarray:
    """Embed every (optionally subsampled) frame and return (N, D) float32."""
    loader_ds: LeRobotDataset | Subset = dataset
    if max_frames is not None and max_frames < len(dataset):
        rng = np.random.default_rng(subsample_seed)
        idx = rng.choice(len(dataset), size=max_frames, replace=False)
        loader_ds = Subset(dataset, sorted(idx.tolist()))
    loader = DataLoader(
        loader_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    embs: list[torch.Tensor] = []
    total = len(loader)
    logging.info(f"{desc}: {total} batches")
    start = time.monotonic()
    for i, batch in enumerate(loader):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        batch = preprocessor(batch)
        with torch.no_grad():
            emb = embed_siglip_pooled(policy, batch)
        embs.append(emb.cpu().float())
        done = i + 1
        if done % 100 == 0 or done == total:
            elapsed = time.monotonic() - start
            eta = elapsed * (total - done) / done
            logging.info(
                f"{desc}: {done}/{total} [elapsed {elapsed:.0f}s, eta {eta:.0f}s]"
            )
    return torch.cat(embs, dim=0).numpy()


def demo_embs_cache_path(
    *,
    cache_dir: str,
    policy_path: str,
    demo_dataset_repo_id: str,
    demo_max_frames: int | None,
    demo_subsample_seed: int,
    demo_rename_map: dict[str, str],
) -> Path:
    """Content-addressed local path for cached demo embeddings."""
    sig_dict = {
        "policy_path": policy_path,
        "demo_dataset_repo_id": demo_dataset_repo_id,
        "demo_max_frames": demo_max_frames,
        "demo_subsample_seed": demo_subsample_seed,
        "demo_rename_map": demo_rename_map,
    }
    sig = hashlib.sha256(json.dumps(sig_dict, sort_keys=True).encode()).hexdigest()[:16]
    return Path(cache_dir) / f"{sig}.safetensors"


def load_or_embed_demos(
    policy: PI05Policy,
    policy_cfg: Any,
    device: torch.device,
    *,
    policy_path: str,
    demo_dataset_repo_id: str,
    demo_max_frames: int | None,
    demo_subsample_seed: int,
    demo_rename_map: dict[str, str],
    batch_size: int,
    num_workers: int,
    cache_dir: str,
) -> np.ndarray:
    """Return demo embeddings, loading from local cache when available."""
    cache_path = demo_embs_cache_path(
        cache_dir=cache_dir,
        policy_path=policy_path,
        demo_dataset_repo_id=demo_dataset_repo_id,
        demo_max_frames=demo_max_frames,
        demo_subsample_seed=demo_subsample_seed,
        demo_rename_map=demo_rename_map,
    )
    if cache_path.is_file():
        demo_embs = load_file(str(cache_path))["embeddings"]
        logging.info(
            f"Loaded cached demo embeddings from {cache_path}: {demo_embs.shape}"
        )
        return demo_embs

    demo_dataset = LeRobotDataset(repo_id=demo_dataset_repo_id, vcodec="auto")
    demo_preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_cfg.pretrained_path),
        dataset_stats=rename_stats(demo_dataset.meta.stats or {}, demo_rename_map),
        preprocessor_overrides={
            "rename_observations_processor": {"rename_map": demo_rename_map},
        },
    )
    demo_embs = embed_dataset(
        policy=policy,
        preprocessor=demo_preprocessor,
        dataset=demo_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        max_frames=demo_max_frames,
        subsample_seed=demo_subsample_seed,
        desc="Embedding demos",
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"embeddings": demo_embs}, str(cache_path))
    logging.info(f"Cached demo embeddings to {cache_path}: {demo_embs.shape}")
    return demo_embs


def knn_distances(
    query: np.ndarray,
    demos: np.ndarray,
    k: int,
    metric: str,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    """Mean distance from each query row to its k nearest demo rows."""
    q = torch.from_numpy(query).to(device, dtype=torch.float32)
    d = torch.from_numpy(demos).to(device, dtype=torch.float32)
    if metric == "cosine":
        q = torch.nn.functional.normalize(q, dim=1)
        d = torch.nn.functional.normalize(d, dim=1)
    out: list[np.ndarray] = []
    for i in range(0, q.shape[0], chunk_size):
        chunk = q[i : i + chunk_size]
        if metric == "l2":
            dist = torch.cdist(chunk, d)
        elif metric == "cosine":
            dist = 1.0 - chunk @ d.T
        else:
            raise ValueError(f"Unknown knn_metric: {metric}")
        top = torch.topk(dist, min(k, dist.shape[1]), dim=1, largest=False).values
        out.append(top.mean(dim=1).cpu().numpy())
    return np.concatenate(out)


def compute_knn_distances_for_dataset(
    dataset: LeRobotDataset,
    policy_path: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    *,
    knn_k: int,
    knn_metric: str,
    knn_chunk_size: int,
    demo_dataset_repo_id: str,
    demo_max_frames: int | None,
    demo_subsample_seed: int,
    demo_rename_map: dict[str, str],
    demo_embs_cache_dir: str,
    frame_indices: list[int] | None = None,
) -> np.ndarray:
    """Return raw mean-of-k-nearest-neighbour distances for the dataset.

    If ``frame_indices`` is provided, only those frames are embedded (e.g. for
    AUROC over a sampled subset). The full ``dataset`` is still used to build
    the policy/preprocessor (which need ``dataset.meta``).
    """
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = Path(policy_path)
    policy_cfg.device = str(device)
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    assert isinstance(policy, PI05Policy)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=str(policy_cfg.pretrained_path)
    )

    loader_ds: LeRobotDataset | Subset = (
        Subset(dataset, frame_indices) if frame_indices is not None else dataset
    )

    try:
        demo_embs = load_or_embed_demos(
            policy=policy,
            policy_cfg=policy_cfg,
            device=device,
            policy_path=policy_path,
            demo_dataset_repo_id=demo_dataset_repo_id,
            demo_max_frames=demo_max_frames,
            demo_subsample_seed=demo_subsample_seed,
            demo_rename_map=demo_rename_map,
            batch_size=batch_size,
            num_workers=num_workers,
            cache_dir=demo_embs_cache_dir,
        )
        rollout_embs = embed_dataset(
            policy=policy,
            preprocessor=preprocessor,
            dataset=loader_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            max_frames=None,
            subsample_seed=0,
            desc="Embedding rollouts",
        )
        logging.info(
            f"Computing kNN distances (k={knn_k}, metric={knn_metric}) "
            f"between {rollout_embs.shape[0]} rollout frames and "
            f"{demo_embs.shape[0]} demo frames..."
        )
        return knn_distances(
            query=rollout_embs,
            demos=demo_embs,
            k=knn_k,
            metric=knn_metric,
            chunk_size=knn_chunk_size,
            device=device,
        )
    finally:
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
