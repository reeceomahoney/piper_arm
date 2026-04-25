"""CLI entry point for policy rollout and dataset collection.

Rolls out the policy in LIBERO, recording observations, actions, and
per-episode success into a LeRobot dataset.
"""

import math
import multiprocessing
import os

os.environ["MUJOCO_GL"] = "egl"

import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import draccus
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts.lerobot_eval import eval_policy as lerobot_eval_policy
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging
from libero.libero import benchmark

multiprocessing.set_start_method("spawn", force=True)


def auto_n_envs(n_episodes: int) -> int:
    """Pick n_envs based on CPU cores, capped by n_episodes."""
    cpu_cores = os.cpu_count() or 4
    by_cpu = max(1, math.floor(cpu_cores * 0.7))
    return min(by_cpu, n_episodes, 64)


@dataclass
class EvalDistConfig:
    policy_path: str = "lerobot/pi05-libero"
    base_dataset_repo_id: str = "lerobot/libero"
    n_episodes: int = 50
    n_envs: int = 10  # 0 = auto-scale based on CPU cores and n_episodes
    dataset_repo_id: str = "reece-omahoney/pi05-libero-10"
    device: str = "cuda"
    max_tasks: int | None = None  # limit number of tasks (None = all)
    seed: int = 0
    use_amp: bool = True


def write_episodes_to_dataset(
    info: dict, dataset: LeRobotDataset, task_desc: str
) -> None:
    """Write episode data returned by lerobot's eval_policy into a LeRobot dataset."""
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
            frame["task"] = task_desc
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

    n_envs = cfg.n_envs if cfg.n_envs > 0 else auto_n_envs(cfg.n_episodes)
    print(f"Using n_envs={n_envs} (requested={cfg.n_envs})")

    # ── Load policy ──
    suite_name = "libero_10"
    env_cfg = LiberoEnvConfig(
        suite_name, fps=10, observation_height=265, observation_width=265
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
        repo_id=cfg.dataset_repo_id,
        fps=int(base_meta.fps),
        features=features,
    )

    # ── Rollout ──
    suite = benchmark.get_benchmark_dict()[suite_name]()  # ty: ignore[unresolved-attribute]
    n_tasks = len(suite.tasks)
    if cfg.max_tasks is not None:
        n_tasks = min(n_tasks, cfg.max_tasks)
    all_successes: list[bool] = []
    t_start = time.monotonic()

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if cfg.use_amp
        else nullcontext()
    )
    try:
        with torch.no_grad(), amp_ctx:
            for task_id in range(n_tasks):
                task_env_cfg = LiberoEnvConfig(
                    suite_name,
                    fps=10,
                    task_ids=[task_id],
                    observation_height=265,
                    observation_width=265,
                )
                task_envs = make_env(task_env_cfg, n_envs=n_envs, use_async_envs=True)
                vec_env = task_envs[suite_name][task_id]
                task_desc = vec_env.call("task_description")[0]  # ty: ignore[unresolved-attribute]
                print(f"\nTask {task_id + 1}/{n_tasks}: {task_desc}")

                info = lerobot_eval_policy(
                    env=vec_env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=cfg.n_episodes,
                    return_episode_data=dataset is not None,
                    start_seed=cfg.seed,
                )

                task_successes = [ep["success"] for ep in info["per_episode"]]
                all_successes.extend(task_successes)
                n_success = sum(task_successes)
                n = len(task_successes)
                print(f"  Success rate: {n_success}/{n} ({100 * n_success / n:.1f}%)")

                write_episodes_to_dataset(info, dataset, task_desc)
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
