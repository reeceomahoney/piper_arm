"""Roll out a base policy in LIBERO, compute value estimates at each step,
and visualize with Rerun (cameras + value over time).

Usage:
    python -m piper_arm.rollout_value_viz \
        --policy-path reece-omahoney/smolvla-libero-16-chunk \
        --value-checkpoint outputs/value/checkpoint_final.pt \
        --n-episodes 1
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
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import inside_slurm
from tqdm import tqdm

from piper_arm.train_value import (  # noqa: F401
    TrainValueConfig,
    load_value_preprocessor,
)
from piper_arm.value_model import ValueConfig, ValueModel


@dataclass
class RolloutValueVizConfig:
    policy_path: str = "reece-omahoney/smolvla-libero-16-chunk"
    value_checkpoint: str = "outputs/value/2026-03-10/11-57-00/checkpoint_40000.pt"
    suite_name: str = "libero_10"
    task_ids: list[int] | None = None
    n_episodes: int = 1
    n_envs: int = 1
    port: int = 9876
    save: bool = False
    output_dir: str = "outputs/rollout_value_viz"


def load_value_model(checkpoint_path: str, device: torch.device) -> ValueModel:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: ValueConfig = ckpt["config"].value
    model = ValueModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def to_hwc_uint8(chw_float: torch.Tensor) -> np.ndarray:
    return (chw_float.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def rollout_with_values(
    policy: SmolVLAPolicy,
    value_model: ValueModel,
    value_preprocessor,
    vec_env,
    preprocessor,
    postprocessor,
    env_preprocessor,
    env_postprocessor,
    task_text: str,
    seed: int,
    device: torch.device,
) -> dict:
    """Run a single episode, collecting observations and value estimates."""
    max_steps = vec_env.call("_max_episode_steps")[0]
    observation, _ = vec_env.reset(seed=[seed])
    policy.reset()

    frames = []
    success = False

    disable_bar = inside_slurm()
    for step in tqdm(
        range(max_steps), desc=f"seed={seed}", leave=False, disable=disable_bar
    ):
        observation = preprocess_observation(observation)
        observation = add_envs_task(vec_env, observation)
        observation = env_preprocessor(observation)
        raw_obs = deepcopy(observation)
        observation = preprocessor(observation)

        with torch.inference_mode():
            action = policy.select_action(observation)

            # Get value estimate from current observation
            obs_i = {
                k: v[0] if isinstance(v, torch.Tensor) else v
                for k, v in raw_obs.items()
            }
            value_batch = value_preprocessor(obs_i)
            logits = value_model(value_batch)
            value = value_model.predict_value(logits).item()

        # Collect frame data
        frame = {"step": step, "value": value}
        for key in sorted(k for k in raw_obs if k.startswith("observation.images.")):
            val = raw_obs[key]
            if isinstance(val, torch.Tensor):
                frame[key] = to_hwc_uint8(val[0])
        frames.append(frame)

        # Step environment
        action = postprocessor(action)
        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action_np = action_transition[ACTION].to("cpu").numpy()

        observation, _, terminated, truncated, info = vec_env.step(action_np)

        if "final_info" in info:
            is_success = info["final_info"].get("is_success")
            if is_success is not None:
                val = is_success[0] if hasattr(is_success, "__len__") else is_success
                if hasattr(val, "item"):
                    val = val.item()
                if val:
                    success = True

        if bool(terminated[0]) or bool(truncated[0]):
            break

    return {"frames": frames, "success": success, "task": task_text}


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
    value_model = load_value_model(cfg.value_checkpoint, device)
    value_preprocessor = load_value_preprocessor(cfg.policy_path)
    print(f"Loaded value model from {cfg.value_checkpoint}")

    # Rollout
    all_results = []
    for task_id, vec_env in envs[cfg.suite_name].items():
        task_text = vec_env.call("task_description")[0]
        print(f"\n=== Task {task_id}: {task_text} ===")

        for ep in range(cfg.n_episodes):
            result = rollout_with_values(
                policy=policy,
                value_model=value_model,
                value_preprocessor=value_preprocessor,
                vec_env=vec_env,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                task_text=task_text,
                seed=ep,
                device=device,
            )
            all_results.append(result)

        vec_env.close()

    # Visualize with Rerun
    rr.init("rollout_value_viz", spawn=True)
    addr = f"rerun+http://127.0.0.1:{cfg.port}/proxy"

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
        rec.connect_grpc(addr)
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
