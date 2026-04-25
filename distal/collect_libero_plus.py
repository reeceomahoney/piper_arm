"""CLI entry point for policy rollout and dataset collection.

Rolls out the policy in LIBERO, recording observations, actions, and
per-episode success into a LeRobot dataset.
"""

import json
import math
import multiprocessing
import os
import random
import re

os.environ["MUJOCO_GL"] = "egl"

import time
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import draccus
import gymnasium as gym
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import _get_suite, _make_env_fns
from lerobot.envs.utils import parse_camera_names
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts.lerobot_eval import eval_policy as lerobot_eval_policy
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

multiprocessing.set_start_method("spawn", force=True)


def auto_parallel_envs() -> int:
    """Default parallel-env count based on CPU cores."""
    cpu_cores = os.cpu_count() or 4
    return max(1, min(64, math.floor(cpu_cores * 0.7)))


@dataclass
class EvalDistConfig:
    policy_path: str = "lerobot/pi05-libero"
    base_dataset_repo_id: str = "lerobot/libero"
    dataset_repo_id: str = "reece-omahoney/pi05-libero-plus"
    device: str = "cuda"
    suites: list[str] = field(
        default_factory=lambda: [
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
        ]
    )
    # Number of tasks rolled out in parallel within one fat AsyncVectorEnv.
    # 0 = auto-scale by CPU cores. Each parallel env is a different task from
    # the same suite, batched into a single GPU policy forward per step.
    parallel_envs: int = 0
    max_tasks: int | None = None  # per-suite task limit (None = all)
    per_cell: int = 1  # tasks sampled per (category, difficulty_level) cell
    seed: int = 0


# Strips LIBERO-plus perturbation suffixes to recover the base task name.
# Applied iteratively since combined perturbations stack (e.g. `_table_3_light_5`).
BASE_TASK_SUFFIX_RE = re.compile(
    r"("
    r"_(?:language|view|light)_[a-z0-9_]+?(?:_noise_\d+)?"
    r"|_(?:table|tb)_\d+"
    r"|_add_\d+"
    r"|_(?:moved_)?level\d+_sample\d+"
    r")+$"
)


def base_task_name(variant_name: str) -> str:
    prev = None
    name = variant_name
    while name != prev:
        prev = name
        name = BASE_TASK_SUFFIX_RE.sub("", name)
    return name


def sample_task_ids(suite_name: str, per_cell: int = 1, seed: int = 0) -> list[int]:
    """Sample task IDs stratified by (base_task, category, difficulty_level)."""
    classif = json.loads(
        (files("libero.libero") / "benchmark" / "task_classification.json").read_text()
    )
    by_cell: dict[tuple, list[int]] = {}
    for entry in classif[suite_name]:
        key = (
            base_task_name(entry["name"]),
            entry["category"],
            entry.get("difficulty_level"),
        )
        by_cell.setdefault(key, []).append(entry["id"])
    rng = random.Random(seed)
    return sorted(
        i for ids in by_cell.values() for i in rng.sample(ids, min(per_cell, len(ids)))
    )


def make_fat_vec_env(
    env_cfg: LiberoEnvConfig, task_ids: list[int]
) -> gym.vector.AsyncVectorEnv:
    """Build one AsyncVectorEnv with a different task_id in each sub-env.

    All task_ids must come from the same suite (max_episode_steps is read off
    the first env, so mixing suites would silently truncate or over-run).
    """
    suite = _get_suite(env_cfg.task)
    cameras = parse_camera_names(env_cfg.camera_name)
    gym_kwargs = dict(env_cfg.gym_kwargs)
    gym_kwargs.pop("task_ids", None)  # not consumed by env factory

    fns: list = []
    for tid in task_ids:
        fns.extend(
            _make_env_fns(
                suite=suite,
                suite_name=env_cfg.task,
                task_id=tid,
                n_envs=1,
                camera_names=cameras,
                episode_length=env_cfg.episode_length,
                init_states=env_cfg.init_states,
                gym_kwargs=gym_kwargs,
                control_mode=env_cfg.control_mode,
                camera_name_mapping=env_cfg.camera_name_mapping,
                is_libero_plus=env_cfg.is_libero_plus,
            )
        )
    return gym.vector.AsyncVectorEnv(fns)


def write_episodes_to_dataset(
    info: dict, dataset: LeRobotDataset, task_descs: list[str]
) -> None:
    """Write episode data returned by lerobot's eval_policy into a LeRobot dataset.

    `task_descs[i]` is the language instruction for env-index `i`. With
    `n_episodes == n_envs` there's one episode per env, so `episode_ix == env_ix`.
    """
    episodes = info["episodes"]
    obs_keys = [k for k in episodes if k.startswith("observation.")]
    features = dataset.meta.features

    for ep_info in info["per_episode"]:
        ep_ix = ep_info["episode_ix"]
        mask = episodes["episode_index"] == ep_ix
        # _compile_episode_data pads one extra copy frame per episode; drop it
        indices = torch.where(mask)[0][:-1]

        for idx in indices:
            frame = {"action": episodes["action"][idx]}
            for key in obs_keys:
                if key not in features:
                    continue
                val = episodes[key][idx]
                if key.startswith("observation.images."):
                    frame[key] = (val.permute(1, 2, 0) * 255).to(torch.uint8)
                else:
                    frame[key] = val
            frame["task"] = task_descs[ep_ix]
            frame["success"] = np.array([ep_info["success"]], dtype=bool)
            dataset.add_frame(frame)
        dataset.save_episode()


@draccus.wrap()
def main(cfg: EvalDistConfig):
    init_logging()
    register_third_party_plugins()

    os.environ["SVT_LOG"] = "1"
    # os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

    device = get_safe_torch_device(cfg.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    parallel_envs = cfg.parallel_envs if cfg.parallel_envs > 0 else auto_parallel_envs()
    print(f"Using parallel_envs={parallel_envs} (requested={cfg.parallel_envs})")

    # ── Load policy ──
    # All LIBERO suites share observation/action specs, so the first suite is
    # a valid representative for building the policy env_cfg.
    env_cfg = LiberoEnvConfig(
        cfg.suites[0],
        fps=20,
        observation_height=256,
        observation_width=256,
        is_libero_plus=True,
    )
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    policy_cfg.pretrained_path = Path(cfg.policy_path)
    policy_cfg.device = str(device)
    policy_cfg.n_action_steps = 10  # ty: ignore[unresolved-attribute]

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
    features = base_meta.features.copy()
    features["success"] = {"dtype": "bool", "shape": (1,), "names": None}
    for key, feat in features.items():
        if key.startswith("observation.images.") and len(feat["shape"]) == 3:
            features[key] = {
                **feat,
                "shape": (env_cfg.observation_height, env_cfg.observation_width, 3),
            }
    dataset = LeRobotDataset.create(
        repo_id=cfg.dataset_repo_id, fps=20, features=features
    )

    # ── Rollout ──
    # Process tasks in suite-homogeneous chunks. Within a chunk, build one fat
    # AsyncVectorEnv where each sub-env is a different task; eval_policy runs
    # one batched rollout, and the policy does a single batched GPU forward
    # per step (no thread-safety issues on its action queue).
    suite_to_ids: dict[str, list[int]] = {}
    for suite in cfg.suites:
        ids = sample_task_ids(suite, per_cell=cfg.per_cell, seed=cfg.seed)
        if cfg.max_tasks is not None:
            ids = ids[: cfg.max_tasks]
        suite_to_ids[suite] = ids

    n_tasks = sum(len(v) for v in suite_to_ids.values())
    n_done = 0
    all_successes: list[bool] = []
    t_start = time.monotonic()

    try:
        with torch.no_grad():
            for suite_name, ids in suite_to_ids.items():
                for chunk_start in range(0, len(ids), parallel_envs):
                    chunk = ids[chunk_start : chunk_start + parallel_envs]

                    chunk_cfg = LiberoEnvConfig(
                        suite_name,
                        fps=20,
                        observation_height=256,
                        observation_width=256,
                        is_libero_plus=True,
                    )
                    vec_env = make_fat_vec_env(chunk_cfg, chunk)
                    task_descs = list(vec_env.call("task_description"))
                    n_done += len(chunk)
                    print(
                        f"\n[{n_done}/{n_tasks}] suite={suite_name} "
                        f"chunk={len(chunk)} tasks (ids {chunk[0]}..{chunk[-1]})"
                    )

                    info = lerobot_eval_policy(
                        env=vec_env,
                        policy=policy,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=len(chunk),
                        return_episode_data=True,
                        start_seed=cfg.seed,
                    )

                    chunk_successes = [ep["success"] for ep in info["per_episode"]]
                    all_successes.extend(chunk_successes)
                    n_success = sum(chunk_successes)
                    n = len(chunk_successes)
                    elapsed = time.monotonic() - t_start
                    overall_pct = 100 * sum(all_successes) / len(all_successes)
                    eta = elapsed / n_done * (n_tasks - n_done) if n_done else 0
                    print(
                        f"  Chunk: {n_success}/{n} ({100 * n_success / n:.1f}%) | "
                        f"Overall: {sum(all_successes)}/{len(all_successes)} "
                        f"({overall_pct:.1f}%) | "
                        f"Elapsed: {elapsed / 60:.1f}min | ETA: {eta / 60:.1f}min"
                    )

                    write_episodes_to_dataset(info, dataset, task_descs)
                    vec_env.close()
    finally:
        dataset.finalize()
        print(
            f"Dataset saved to {dataset.root}"
            f" ({dataset.num_episodes} episodes, {dataset.num_frames} frames)"
        )
        dataset.push_to_hub()

    elapsed = time.monotonic() - t_start
    total_successes = sum(all_successes)
    n = len(all_successes)
    pct = 100 * total_successes / n
    print(f"\nOverall success rate: {total_successes}/{n} ({pct:.1f}%)")
    print(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f}min)")


if __name__ == "__main__":
    main()
