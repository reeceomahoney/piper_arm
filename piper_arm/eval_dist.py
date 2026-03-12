"""CLI entry point for rollout trace capture with Mahalanobis OOD detection.

Orchestrates two phases:
  1. Fit a Gaussian over VLM prefix embeddings (or load cached stats).
  2. Roll out the policy in LIBERO, recording observations, actions,
     and per-timestep Mahalanobis distances into a LeRobot dataset.

Core logic lives in sibling modules:
  - embedding.py   — VLM prefix extraction (PI05 / SmolVLA)
  - mahalanobis.py  — Gaussian fitting and distance computation
  - rollout.py      — environment rollout loop and plotting

Usage:
    python -m piper_arm.eval_dist \
        --policy-path lerobot/pi05_libero_finetuned \
        --base-dataset-repo-id reece-omahoney/libero \
        --load-stats outputs/eval_dist/.../gauss_stats.npz \
        --n-episodes 1
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import draccus
import numpy as np
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from libero.libero import benchmark
from tqdm import tqdm

from piper_arm.mahalanobis import fit_gaussian_from_dataset
from piper_arm.rollout import build_frame, plot_traces, rollout


@dataclass
class EvalDistConfig:
    policy_path: str = "reece-omahoney/smolvla-libero-16-chunk"
    base_dataset_repo_id: str = "reece-omahoney/libero"
    n_episodes: int = 50
    n_envs: int = 16
    batch_size: int = 32
    num_workers: int = 8
    load_stats: str | None = "outputs/eval_dist/2026-03-02/13-11-13/gauss_stats.npz"
    dataset_repo_id: str | None = "reece-omahoney/libero-10"
    output_dir: str = "outputs/eval_dist"
    device: str = "cuda:1"


@draccus.wrap()
def main(cfg: EvalDistConfig):
    os.environ["SVT_LOG"] = "1"
    # ── Load policy ──
    suite_name = "libero_10"
    env_cfg = LiberoEnvConfig(suite_name, fps=10)
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    policy_cfg.pretrained_path = Path(cfg.policy_path)
    policy_cfg.device = cfg.device

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    assert isinstance(policy, (PI05Policy, SmolVLAPolicy))
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=str(policy_cfg.pretrained_path)
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg, policy_cfg
    )
    base_dataset = LeRobotDataset(repo_id=cfg.base_dataset_repo_id)

    # ── Phase 1: Get Gaussian stats ──
    if cfg.load_stats is not None:
        print(f"Loading cached stats from {cfg.load_stats}")
        data = np.load(cfg.load_stats)
        gauss_mean = data["mean"]
        gauss_cov_inv = data["cov_inv"]
        print(f"Loaded Gaussian stats, dim={gauss_mean.shape[0]}")
    else:
        gauss_mean, gauss_cov_inv = fit_gaussian_from_dataset(
            policy=policy,
            preprocessor=preprocessor,
            dataset=base_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )

    # ── Output directory ──
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = Path(cfg.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a "latest" symlink for convenience
    latest_link = Path(cfg.output_dir) / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(output_dir.resolve())

    # Save Gaussian stats for reuse
    np.savez(output_dir / "gauss_stats.npz", mean=gauss_mean, cov_inv=gauss_cov_inv)
    print(f"Saved Gaussian stats to {output_dir / 'gauss_stats.npz'}")

    # ── Dataset setup ──
    dataset = None
    if cfg.dataset_repo_id:
        features = base_dataset.meta.features.copy()
        features["maha_distance"] = {"dtype": "float32", "shape": (1,), "names": None}
        features["steps_remaining"] = {"dtype": "int32", "shape": (1,), "names": None}
        features["success"] = {"dtype": "bool", "shape": (1,), "names": None}
        dataset = LeRobotDataset.create(
            repo_id=cfg.dataset_repo_id,
            fps=int(base_dataset.meta.fps),
            features=features,
            image_writer_threads=4 * len(base_dataset.meta.camera_keys),
        )

    # ── Phase 2: Rollout with capture ──
    suite = benchmark.get_benchmark_dict()[suite_name]()
    n_tasks = len(suite.tasks)
    all_results: list[dict[str, Any]] = []
    t_start = time.monotonic()

    try:
        for task_id in range(n_tasks):
            task_env_cfg = LiberoEnvConfig(suite_name, fps=10, task_ids=[task_id])
            task_envs = make_env(task_env_cfg, n_envs=cfg.n_envs)
            vec_env = task_envs[suite_name][task_id]
            task_desc = vec_env.call("task_description")[0]  # type: ignore[attr-defined]

            ep = 0
            task_results: list[dict[str, Any]] = []
            pbar = tqdm(
                total=cfg.n_episodes, desc=f"Task {task_id + 1}/{n_tasks}", unit="ep"
            )
            while ep < cfg.n_episodes:
                batch_seeds = list(range(ep, ep + cfg.n_envs))
                batch_results = rollout(
                    policy=policy,
                    vec_env=vec_env,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    gauss_mean=gauss_mean,
                    gauss_cov_inv=gauss_cov_inv,
                    seeds=batch_seeds,
                    desc=f"  Ep {ep}-{ep + cfg.n_envs - 1}",
                )

                for i, result in enumerate(batch_results):
                    ep_num = ep + i
                    if ep_num >= cfg.n_episodes:
                        break

                    # Build LeRobot dataset incrementally
                    if dataset is not None:
                        n_frames = len(result["observations"])
                        for frame_idx in range(n_frames):
                            frame = build_frame(
                                result["observations"][frame_idx],
                                result["actions"][frame_idx],
                                dataset.meta.features,
                            )
                            frame["task"] = task_desc
                            frame["steps_remaining"] = np.array(
                                [n_frames - frame_idx - 1], dtype=np.int32
                            )
                            frame["success"] = np.array([result["success"]], dtype=bool)
                            dataset.add_frame(frame)
                        dataset.save_episode()

                    pbar.update(1)
                    task_results.append(result)
                    all_results.append(result)

                ep += cfg.n_envs
            pbar.close()

            task_successes = sum(r["success"] for r in task_results)
            n = len(task_results)
            pct = 100 * task_successes / n
            print(f"  Success rate: {task_successes}/{n} ({pct:.1f}%)")

            vec_env.close()
    finally:
        if dataset is not None:
            dataset.finalize()
            print(
                f"Dataset saved to {dataset.root}"
                f" ({dataset.num_episodes} episodes, {dataset.num_frames} frames)"
            )

    elapsed = time.monotonic() - t_start
    total_successes = sum(r["success"] for r in all_results)
    n = len(all_results)
    pct = 100 * total_successes / n
    print(f"\nOverall success rate: {total_successes}/{n} ({pct:.1f}%)")
    print(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    print(f"Outputs saved to {output_dir}")

    # ── Plot ──
    plot_traces(all_results, output_dir)


if __name__ == "__main__":
    main()
