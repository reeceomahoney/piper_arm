"""Shared LIBERO sim-eval helper used by training and standalone eval.

Builds fat ``AsyncVectorEnv``s over LIBERO task IDs, runs
``lerobot_eval_policy`` per chunk, and returns per-suite + overall metrics.

Two modes, selected by ``is_libero_plus``:

- libero-plus: each chunk packs up to ``parallel_envs`` distinct task IDs
  (1 env each) into one vec env. Task IDs come from
  ``sample_task_ids(per_cell, task_seed)`` and match the rollout dataset
  whenever those knobs match collect's settings.
- base LIBERO: 10 tasks per suite, each task gets its own vec env with
  ``n_envs_per_task`` sub-envs (distinct ``episode_index`` → distinct init
  states).
"""

import json
import logging
import time
from collections import defaultdict
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
from lerobot.envs.configs import LiberoEnv
from lerobot.scripts.lerobot_eval import eval_policy as lerobot_eval_policy

from distal.collect_libero_plus import (
    auto_parallel_envs,
    base_task_name,
    make_fat_vec_env,
    sample_task_ids,
)


def resolve_eval_task_ids(
    suite_name: str,
    per_cell: int,
    task_seed: int,
    base_task: str | None = None,
    max_tasks: int | None = None,
) -> list[int]:
    """Sample task IDs for a suite, optionally filtered to one base task."""
    ids = sample_task_ids(suite_name, per_cell=per_cell, seed=task_seed)
    if base_task is not None:
        classif = json.loads(
            (
                files("libero.libero") / "benchmark" / "task_classification.json"
            ).read_text()
        )
        id_to_base = {e["id"]: base_task_name(e["name"]) for e in classif[suite_name]}
        ids = [i for i in ids if id_to_base[i] == base_task]
        if not ids:
            raise ValueError(
                f"base_task={base_task!r} matched no task IDs in suite {suite_name!r}"
            )
    if max_tasks is not None:
        ids = ids[:max_tasks]
    return ids


def run_sim_eval(
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    suites: list[str],
    is_libero_plus: bool,
    fps: int,
    observation_height: int,
    observation_width: int,
    per_cell: int,
    task_seed: int,
    base_task: str | None,
    max_tasks: int | None,
    parallel_envs: int,
    n_envs_per_task: int,
    n_episodes_per_task: int,
    seed: int,
    videos_dir: Path | None = None,
    max_episodes_rendered: int = 4,
    wandb_run=None,
    wandb_step: int | None = None,
) -> dict[str, float]:
    """Run LIBERO sim eval and return per-suite + overall metrics."""
    parallel_envs = parallel_envs if parallel_envs > 0 else auto_parallel_envs()
    logging.info(
        f"Running sim eval (is_libero_plus={is_libero_plus}, "
        f"parallel_envs={parallel_envs}, n_envs_per_task={n_envs_per_task}, "
        f"n_ep_per_task={n_episodes_per_task})"
    )
    policy.eval()

    suite_metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"successes": [], "sum_rewards": []}
    )
    rendered = 0
    first_video_path: str | None = None
    t0 = time.monotonic()

    with torch.no_grad():
        for suite_name in suites:
            if is_libero_plus:
                ids = resolve_eval_task_ids(
                    suite_name, per_cell, task_seed, base_task, max_tasks
                )
                chunks = [
                    ids[i : i + parallel_envs]
                    for i in range(0, len(ids), parallel_envs)
                ]
                envs_per_task = 1
            else:
                ids = list(range(10))
                if max_tasks is not None:
                    ids = ids[:max_tasks]
                chunks = [[tid] for tid in ids]
                envs_per_task = (
                    n_envs_per_task if n_envs_per_task > 0 else parallel_envs
                )

            env_cfg = LiberoEnv(
                task=suite_name,
                fps=fps,
                observation_height=observation_height,
                observation_width=observation_width,
                is_libero_plus=is_libero_plus,
            )

            for chunk_idx, chunk in enumerate(chunks):
                vec_env = make_fat_vec_env(
                    env_cfg, chunk, n_envs_per_task=envs_per_task
                )
                try:
                    chunk_videos_dir: Path | None = None
                    chunk_max_render = 0
                    if videos_dir is not None and rendered < max_episodes_rendered:
                        chunk_videos_dir = videos_dir / f"{suite_name}_{chunk_idx}"
                        chunk_max_render = max_episodes_rendered - rendered

                    info = lerobot_eval_policy(
                        env=vec_env,
                        policy=policy,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=(len(chunk) * envs_per_task * n_episodes_per_task),
                        max_episodes_rendered=chunk_max_render,
                        videos_dir=chunk_videos_dir,
                        start_seed=seed,
                    )
                    for ep in info["per_episode"]:
                        suite_metrics[suite_name]["successes"].append(
                            float(ep["success"])
                        )
                        suite_metrics[suite_name]["sum_rewards"].append(
                            float(ep["sum_reward"])
                        )
                    chunk_videos = info.get("video_paths", [])
                    if first_video_path is None and chunk_videos:
                        first_video_path = chunk_videos[0]
                    rendered += len(chunk_videos)
                finally:
                    vec_env.close()

    eval_s = time.monotonic() - t0

    overall_succ = [s for m in suite_metrics.values() for s in m["successes"]]
    overall_rew = [r for m in suite_metrics.values() for r in m["sum_rewards"]]
    pc_success = float(np.mean(overall_succ) * 100) if overall_succ else float("nan")
    avg_sum_reward = float(np.mean(overall_rew)) if overall_rew else float("nan")

    metrics: dict[str, float] = {
        "eval_s": eval_s,
        "avg_sum_reward": avg_sum_reward,
        "pc_success": pc_success,
    }
    for suite_name, m in suite_metrics.items():
        suite_succ = (
            float(np.mean(m["successes"]) * 100) if m["successes"] else float("nan")
        )
        suite_rew = (
            float(np.mean(m["sum_rewards"])) if m["sum_rewards"] else float("nan")
        )
        metrics[f"pc_success_{suite_name}"] = suite_succ
        metrics[f"avg_sum_reward_{suite_name}"] = suite_rew
        logging.info(
            f"Suite {suite_name}: pc_success={suite_succ:.1f}% "
            f"avg_sum_reward={suite_rew:.3f} (n={len(m['successes'])})"
        )
    logging.info(
        f"Overall: pc_success={pc_success:.1f}% avg_sum_reward={avg_sum_reward:.3f} "
        f"(n={len(overall_succ)}) eval_s={eval_s:.1f}"
    )

    if wandb_run is not None and first_video_path is not None:
        import wandb

        wandb_run.log(
            {"eval/video": wandb.Video(str(first_video_path), fps=fps, format="mp4")},
            step=wandb_step,
        )

    return metrics
