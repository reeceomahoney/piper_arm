"""CLI entry point for policy rollout and dataset collection.

Rolls out the policy in LIBERO, recording observations, actions, and
per-episode success into a LeRobot dataset.
"""

import multiprocessing
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import draccus
import numpy as np
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from libero.libero import benchmark
from tqdm import tqdm

from piper_arm.rollout import build_frame, rollout

multiprocessing.set_start_method("spawn", force=True)


@dataclass
class EvalDistConfig:
    policy_path: str = "reece-omahoney/smolvla-libero-16-chunk"
    base_dataset_repo_id: str = "lerobot/libero"
    n_episodes: int = 50
    n_envs: int = 50
    dataset_repo_id: str | None = "reece-omahoney/libero-10"
    device: str = "cuda:1"
    max_tasks: int | None = None  # limit number of tasks (None = all)


@draccus.wrap()
def main(cfg: EvalDistConfig):
    os.environ["SVT_LOG"] = "1"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
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
    base_meta = LeRobotDatasetMetadata(repo_id=cfg.base_dataset_repo_id)

    # ── Dataset setup ──
    dataset = None
    if cfg.dataset_repo_id:
        features = base_meta.features.copy()
        features["success"] = {"dtype": "bool", "shape": (1,), "names": None}
        dataset = LeRobotDataset.create(
            repo_id=cfg.dataset_repo_id,
            fps=int(base_meta.fps),
            features=features,
            image_writer_threads=8 * len(base_meta.camera_keys),
            vcodec="auto",
        )

    # ── Rollout ──
    suite = benchmark.get_benchmark_dict()[suite_name]()
    n_tasks = len(suite.tasks)
    if cfg.max_tasks is not None:
        n_tasks = min(n_tasks, cfg.max_tasks)
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
                    seeds=batch_seeds,
                    desc=f"  Ep {ep}-{ep + cfg.n_envs - 1}",
                )

                for i, result in enumerate(batch_results):
                    ep_num = ep + i
                    if ep_num >= cfg.n_episodes:
                        break

                    if dataset is not None:
                        n_frames = len(result["observations"])
                        for frame_idx in range(n_frames):
                            frame = build_frame(
                                result["observations"][frame_idx],
                                result["actions"][frame_idx],
                                dataset.meta.features,
                            )
                            frame["task"] = task_desc
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
            dataset.push_to_hub()

    elapsed = time.monotonic() - t_start
    total_successes = sum(r["success"] for r in all_results)
    n = len(all_results)
    pct = 100 * total_successes / n
    print(f"\nOverall success rate: {total_successes}/{n} ({pct:.1f}%)")
    print(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f}min)")


if __name__ == "__main__":
    main()
