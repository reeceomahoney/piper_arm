"""Roll out a base policy in LIBERO, compute value estimates at each step,
and visualize with Rerun (cameras + value over time).
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import draccus
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import inside_slurm
from tqdm import tqdm

from distal.value_model import RECAPValueNetwork


@dataclass
class RolloutValueVizConfig:
    policy_path: str = "reece-omahoney/adv-libero-base"
    value_checkpoint: str = "reece-omahoney/value-success-expert"
    suite_name: str = "libero_10"
    task_ids: list[int] | None = None
    n_envs: int = 5
    save: bool = False
    output_dir: str = "outputs/rollout_value_viz"


def to_hwc_uint8(chw_float: torch.Tensor) -> np.ndarray:
    return (chw_float.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def add_task(observation: dict, vec_env) -> dict:
    try:
        observation["task"] = list(vec_env.call("task_description"))
    except (AttributeError, NotImplementedError):
        try:
            observation["task"] = list(vec_env.call("task"))
        except (AttributeError, NotImplementedError):
            observation["task"] = [""] * vec_env.num_envs
    return observation


def rollout_with_values(
    policy: SmolVLAPolicy,
    value_model: RECAPValueNetwork,
    value_preprocessor,
    vec_env,
    preprocessor,
    postprocessor,
    env_preprocessor,
    env_postprocessor,
    task_text: str,
    seed: int,
) -> list[dict]:
    """Run episodes across all envs, collecting observations and value estimates."""
    n_envs = vec_env.num_envs
    max_steps = vec_env.call("_max_episode_steps")[0]
    observation, _ = vec_env.reset(seed=[seed + i for i in range(n_envs)])
    policy.reset()

    frames = [[] for _ in range(n_envs)]
    successes = [False] * n_envs
    dones = [False] * n_envs

    disable_bar = inside_slurm()
    for step in tqdm(
        range(max_steps), desc=f"seed={seed}", leave=False, disable=disable_bar
    ):
        observation = preprocess_observation(observation)
        observation = add_task(observation, vec_env)
        observation = env_preprocessor(observation)
        raw_obs = deepcopy(observation)
        observation = preprocessor(observation)

        with torch.inference_mode():
            action = policy.select_action(observation)

            # Get value estimates for all envs
            value_batch = value_preprocessor(raw_obs)
            values = value_model.predict_value(value_batch)

        # Collect frame data per env
        for ei in range(n_envs):
            if dones[ei]:
                continue
            frame = {"step": step, "value": values[ei].item()}
            for key in sorted(
                k for k in raw_obs if k.startswith("observation.images.")
            ):
                val = raw_obs[key]
                if isinstance(val, torch.Tensor):
                    frame[key] = to_hwc_uint8(val[ei])
            frames[ei].append(frame)

        # Step environment
        action = postprocessor(action)
        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action_np = action_transition[ACTION].to("cpu").numpy()

        observation, _, terminated, truncated, info = vec_env.step(action_np)

        if info["done"].any():
            for ei in range(n_envs):
                if info["done"][ei] and info["is_success"][ei]:
                    successes[ei] = True

        for ei in range(n_envs):
            if bool(terminated[ei]) or bool(truncated[ei]):
                dones[ei] = True

        if all(dones):
            break

    return [
        {"frames": frames[ei], "success": successes[ei], "task": task_text}
        for ei in range(n_envs)
    ]


def log_episode_to_rerun(result: dict, episode_idx: int) -> None:
    """Log a single episode's data to the active Rerun recording."""
    frames = result["frames"]
    status = "success" if result["success"] else "failure"
    task = result["task"]
    print(f"Episode {episode_idx} ({status}): {len(frames)} steps, task={task}")

    for frame in frames:
        rr.set_time("step", sequence=frame["step"])

        for key in sorted(k for k in frame if k.startswith("observation.images.")):
            cam_name = key.replace("observation.images.", "cameras/")
            rr.log(cam_name, rr.Image(frame[key]))

        rr.log("metrics/value", rr.Scalars(frame["value"]))

    rr.log("episode/success", rr.TextDocument(status))


@draccus.wrap()
def main(cfg: RolloutValueVizConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load policy
    env_cfg = LiberoEnvConfig(cfg.suite_name, fps=10, task_ids=cfg.task_ids or [9])
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    policy_cfg.pretrained_path = Path(cfg.policy_path)

    envs = make_env(env_cfg, n_envs=cfg.n_envs)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    assert isinstance(policy, SmolVLAPolicy)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=str(policy_cfg.pretrained_path)
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg, policy_cfg
    )

    # Load value model & value preprocessor
    value_model = RECAPValueNetwork.from_pretrained(cfg.value_checkpoint)
    value_model.to(device).eval()
    value_preprocessor = PolicyProcessorPipeline.from_pretrained(
        cfg.value_checkpoint, config_filename="policy_preprocessor.json"
    )
    print(f"Loaded value model from {cfg.value_checkpoint}")

    # Rollout
    all_results = []
    for task_id, vec_env in envs[cfg.suite_name].items():
        task_text = vec_env.call("task_description")[0]  # ty: ignore[unresolved-attribute]
        print(f"\n=== Task {task_id}: {task_text} ===")

        results = rollout_with_values(
            policy=policy,
            value_model=value_model,
            value_preprocessor=value_preprocessor,
            vec_env=vec_env,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            task_text=task_text,
            seed=0,
        )
        all_results.extend(results)

        vec_env.close()

    # Visualize with Rerun
    rr.init("rollout_value_viz", spawn=True)

    for idx, result in enumerate(all_results):
        status = "ok" if result["success"] else "fail"

        # Build blueprint from the camera keys in the first frame
        cam_keys = sorted(
            k for k in result["frames"][0] if k.startswith("observation.images.")
        )
        cam_views = [
            rrb.Spatial2DView(
                origin=k.replace("observation.images.", "cameras/"),
            )
            for k in cam_keys
        ]
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(*cam_views), rrb.TimeSeriesView(origin="metrics/")
            ),
        )

        rec = rr.RecordingStream(application_id=f"episode_{idx}_{status}")
        rec.connect_grpc()
        with rec:
            rr.send_blueprint(blueprint)
            log_episode_to_rerun(result, idx)

    if cfg.save:
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rrd_path = output_dir / "rollout_value_viz.rrd"
        rr.save(rrd_path)
        print(f"Saved .rrd to {rrd_path}")

    print("\nRerun viewer launched. Close the viewer window to exit.")


if __name__ == "__main__":
    main()
