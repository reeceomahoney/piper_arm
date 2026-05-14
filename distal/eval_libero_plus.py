"""Standalone LIBERO-plus evaluation.

Mirrors the sim-eval env setup used inside ``distal.train_pi_star`` so a
trained checkpoint can be re-evaluated without re-running the training loop.
Delegates rollout logic to ``distal.sim_eval.run_libero_plus_eval``; this
script only handles policy loading, preprocessor wiring, and writing the
``summary.json`` artifact.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

# isort: off
# Need this to prevent lib errors
import wand.api  # noqa: F401
# isort: on

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

from distal.sim_eval import LiberoPlusEvalConfig, run_libero_plus_eval


@dataclass
class EvalLiberoPlusConfig:
    policy_path: str = "reece-omahoney/pistar-knn-rel-libero-plus"
    n_action_steps: int = 10
    cfg_beta: float | None = None
    device: str = "cuda"
    seed: int = 42

    eval: LiberoPlusEvalConfig = field(
        default_factory=lambda: LiberoPlusEvalConfig(
            base_task="turn_on_the_stove", parallel_envs=25
        )
    )

    output_dir: str | None = None
    max_episodes_rendered: int = 0


def log_group_table(
    metrics: dict[str, float], *, prefix: str, header: str, title: str
) -> None:
    names = sorted(
        k[len(f"pc_success_{prefix}_") :]
        for k in metrics
        if k.startswith(f"pc_success_{prefix}_")
    )
    if not names:
        return
    name_w = max(len(header), max(len(n) for n in names))
    logging.info(f"{title}:")
    logging.info(
        f"  {header:<{name_w}}  {'pc_success':>10}  {'avg_sum_reward':>14}  {'n':>4}"
    )
    for name in names:
        pc = metrics[f"pc_success_{prefix}_{name}"]
        rew = metrics[f"avg_sum_reward_{prefix}_{name}"]
        n = int(metrics[f"n_{prefix}_{name}"])
        logging.info(f"  {name:<{name_w}}  {pc:>9.1f}%  {rew:>14.3f}  {n:>4}")


@draccus.wrap()
def main(cfg: EvalLiberoPlusConfig):
    init_logging()
    register_third_party_plugins()

    device = get_safe_torch_device(cfg.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    rep_env_cfg = LiberoEnv(
        task=cfg.eval.suites[0],
        fps=cfg.eval.fps,
        observation_height=cfg.eval.observation_height,
        observation_width=cfg.eval.observation_width,
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

    metrics = run_libero_plus_eval(
        cfg.eval,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
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

    log_group_table(
        metrics, prefix="base", header="base_task", title="Per-task results"
    )
    log_group_table(
        metrics,
        prefix="cat",
        header="perturbation_category",
        title="Per-perturbation-category results",
    )


if __name__ == "__main__":
    main()
