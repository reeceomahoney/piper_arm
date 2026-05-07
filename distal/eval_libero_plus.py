"""Standalone LIBERO-plus evaluation.

Mirrors the sim-eval env setup used inside ``distal.train_pi_star`` so a
trained checkpoint can be re-evaluated without re-running the training loop.
Delegates all rollout logic to ``distal.sim_eval.run_sim_eval``; this script
only handles policy loading, preprocessor wiring, and writing the
``summary.json`` artifact.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import draccus
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

from distal.sim_eval import run_sim_eval


@dataclass
class EvalLiberoPlusConfig:
    policy_path: str = "reece-omahoney/pistar06-libero-plus-maha"
    n_action_steps: int = 10
    cfg_beta: float | None = None
    device: str = "cuda"
    seed: int = 42

    # Env knobs (must match the values used at collect time so eval rolls out
    # the same task IDs that appear in the rollout dataset).
    suites: list[str] = field(default_factory=lambda: ["libero_goal"])
    fps: int = 20
    observation_height: int = 256
    observation_width: int = 256
    per_cell: int = 1
    task_seed: int = 0

    # Optional task-set restrictions.
    base_task: str | None = "turn_on_the_stove"
    max_tasks: int | None = None

    # Parallelism: 0 = auto-scale by CPU cores. Each chunk packs up to
    # ``parallel_envs`` distinct task IDs into one fat vec env.
    parallel_envs: int = 0
    n_episodes_per_task: int = 1

    output_dir: str | None = None
    max_episodes_rendered: int = 4


@draccus.wrap()
def main(cfg: EvalLiberoPlusConfig):
    init_logging()
    register_third_party_plugins()

    device = get_safe_torch_device(cfg.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    rep_env_cfg = LiberoEnv(
        task=cfg.suites[0],
        fps=cfg.fps,
        observation_height=cfg.observation_height,
        observation_width=cfg.observation_width,
        is_libero_plus=True,
    )

    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    policy_cfg.pretrained_path = Path(cfg.policy_path)
    policy_cfg.device = str(device)
    policy_cfg.n_action_steps = cfg.n_action_steps  # ty: ignore[unresolved-attribute]
    if cfg.cfg_beta is not None:
        policy_cfg.cfg_beta = cfg.cfg_beta  # ty: ignore[unresolved-attribute]

    policy = make_policy(cfg=policy_cfg, env_cfg=rep_env_cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=str(policy_cfg.pretrained_path)
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=rep_env_cfg, policy_cfg=policy_cfg
    )

    output_dir = (
        Path(cfg.output_dir)
        if cfg.output_dir
        else Path("outputs/eval_libero_plus") / time.strftime("%Y-%m-%d/%H-%M-%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output dir: {output_dir}")

    metrics = run_sim_eval(
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        suites=cfg.suites,
        is_libero_plus=True,
        fps=cfg.fps,
        observation_height=cfg.observation_height,
        observation_width=cfg.observation_width,
        per_cell=cfg.per_cell,
        task_seed=cfg.task_seed,
        base_task=cfg.base_task,
        max_tasks=cfg.max_tasks,
        parallel_envs=cfg.parallel_envs,
        n_envs_per_task=1,
        n_episodes_per_task=cfg.n_episodes_per_task,
        seed=cfg.seed,
        videos_dir=output_dir / "videos",
        max_episodes_rendered=cfg.max_episodes_rendered,
    )

    summary = {
        "policy_path": cfg.policy_path,
        "cfg_beta": cfg.cfg_beta,
        **metrics,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Saved summary to {summary_path}")

    base_names = sorted(
        k[len("pc_success_base_") :]
        for k in metrics
        if k.startswith("pc_success_base_")
    )
    if base_names:
        name_w = max(len("base_task"), max(len(b) for b in base_names))
        logging.info("Per-base-task results:")
        logging.info(
            f"  {'base_task':<{name_w}}  {'pc_success':>10}  "
            f"{'avg_sum_reward':>14}  {'n':>4}"
        )
        for base in base_names:
            pc = metrics[f"pc_success_base_{base}"]
            rew = metrics[f"avg_sum_reward_base_{base}"]
            n = int(metrics[f"n_base_{base}"])
            logging.info(f"  {base:<{name_w}}  {pc:>9.1f}%  {rew:>14.3f}  {n:>4}")


if __name__ == "__main__":
    main()
