"""
Generate a dataset of policy rollouts in LIBERO with contact states.

Rolls out a pretrained SmolVLA checkpoint in the libero_object environment,
recording observations, actions, gripper contact states, and object poses
at every frame.
"""

import os
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv, _get_suite
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION

HF_USER = "reece-omahoney"
PRETRAINED_PATH = "reece-omahoney/smolvla-libero"
SUITE_NAME = "libero_object"
REPO_ID = f"{HF_USER}/libero-affordances"
N_EPISODES_PER_TASK = 1
FPS = 30
DEVICE_ID = 0


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (x,y,z,w) to axis-angle (3,)."""
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(1.0 - w * w, 0.0))
    if den > 1e-10:
        angle = 2.0 * np.arccos(w)
        axis = quat[:3] / den
        return (axis * angle).astype(np.float32)
    return np.zeros(3, dtype=np.float32)


def flatten_state(obs: dict) -> np.ndarray:
    """Flatten robot_state to 8D: eef_pos(3) + axis_angle(3) + gripper_qpos(2)."""
    eef_pos = obs["robot_state"]["eef"]["pos"]
    eef_quat = obs["robot_state"]["eef"]["quat"]
    gripper_qpos = obs["robot_state"]["gripper"]["qpos"]
    axis_angle = quat_to_axis_angle(eef_quat)
    return np.concatenate([eef_pos, axis_angle, gripper_qpos]).astype(np.float32)


def check_gripper_contact(env: LiberoEnv) -> bool:
    """Check if the gripper is in contact with any non-gripper geom."""
    robosuite_env = env._env.env
    gripper = robosuite_env.robots[0].gripper
    contacts = robosuite_env.get_contacts(gripper)
    return len(contacts) > 0


def get_object_position(env: LiberoEnv) -> np.ndarray:
    """Get the current position of the object of interest."""
    robosuite_env = env._env.env
    object_name = robosuite_env.obj_of_interest[0]
    return (
        robosuite_env.sim.data.body_xpos[robosuite_env.obj_body_id[object_name]]
        .copy()
        .astype(np.float32)
    )


def add_batch_dim(obs: dict) -> dict:
    """Recursively add a leading batch dimension to all numpy arrays."""
    result = {}
    for k, v in obs.items():
        if isinstance(v, dict):
            result[k] = add_batch_dim(v)
        elif isinstance(v, np.ndarray):
            result[k] = v[np.newaxis]
        else:
            result[k] = v
    return result


def main():
    os.environ["MUJOCO_GL"] = "egl"
    if torch.cuda.device_count() > 1:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = "1" if DEVICE_ID == 0 else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

    # Load policy
    env_cfg = LiberoEnvConfig("libero_object")
    policy_cfg = PreTrainedConfig.from_pretrained(PRETRAINED_PATH)
    policy_cfg.pretrained_path = Path(PRETRAINED_PATH)

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=policy_cfg.pretrained_path
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg, policy_cfg
    )

    # Create dataset
    features = {
        "observation.images.image": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": None,
        },
        "observation.images.image2": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": None,
        },
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
        "contact_state": {"dtype": "float32", "shape": (1,), "names": None},
        "object_pos": {"dtype": "float32", "shape": (3,), "names": None},
        "gripper_pos": {"dtype": "float32", "shape": (3,), "names": None},
        "gripper_to_obj_dist": {"dtype": "float32", "shape": (1,), "names": None},
    }

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=features,
        use_videos=True,
    )

    suite = _get_suite(SUITE_NAME)
    n_tasks = len(suite.tasks)
    max_steps = TASK_SUITE_MAX_STEPS.get(SUITE_NAME, 280)

    for task_id in range(n_tasks):
        task = suite.get_task(task_id)
        print(f"\n=== Task {task_id}/{n_tasks}: {task.language} ===")

        # Create env once per task (reuse across episodes)
        env = LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=SUITE_NAME,
            obs_type="pixels_agent_pos",
            camera_name="agentview_image,robot0_eye_in_hand_image",
            init_states=True,
            episode_index=0,
            control_mode="relative",
        )

        for ep in range(N_EPISODES_PER_TASK):
            env._init_state_id = ep
            obs, info = env.reset()
            policy.reset()
            is_success = False

            for step in range(max_steps):
                # Check contact (matches current obs/sim state)
                contact = check_gripper_contact(env)

                # Process observation for policy
                batched_obs = add_batch_dim(obs)
                policy_obs = preprocess_observation(batched_obs)
                policy_obs["task"] = [env.task_description]
                policy_obs = env_preprocessor(policy_obs)
                policy_obs = preprocessor(policy_obs)

                with torch.inference_mode():
                    action = policy.select_action(policy_obs)
                action = postprocessor(action)

                action_transition = {ACTION: action}
                action_transition = env_postprocessor(action_transition)
                action_np = action_transition[ACTION].squeeze(0).cpu().numpy()

                # Step the underlying env directly (bypass auto-reset)
                raw_obs, reward, done, step_info = env._env.step(action_np)
                is_success = env._env.check_success()
                terminated = done or is_success

                # Store images flipped 180° to match training data orientation
                img1 = obs["pixels"]["image"][::-1, ::-1].copy()
                img2 = obs["pixels"]["image2"][::-1, ::-1].copy()
                state = flatten_state(obs)

                object_pos = get_object_position(env)
                gripper_pos = obs["robot_state"]["eef"]["pos"].astype(np.float32)
                dist = np.linalg.norm(gripper_pos - object_pos)

                frame = {
                    "task": env.task_description,
                    "observation.images.image": img1,
                    "observation.images.image2": img2,
                    "observation.state": state,
                    "action": action_np.astype(np.float32),
                    "next.reward": np.array([reward], dtype=np.float32),
                    "next.done": np.array([terminated], dtype=bool),
                    "next.success": np.array([is_success], dtype=bool),
                    "contact_state": np.array([float(contact)], dtype=np.float32),
                    "object_pos": object_pos,
                    "gripper_pos": gripper_pos,
                    "gripper_to_obj_dist": np.array([dist], dtype=np.float32),
                }
                dataset.add_frame(frame)

                if terminated:
                    break

                # Get next observation
                obs = env._format_raw_obs(raw_obs)

            dataset.save_episode()
            print(f"  Episode {ep}: {step + 1} steps, success={is_success}")

        env.close()

    dataset.finalize()
    print(f"\nDataset saved to {dataset.root}")


if __name__ == "__main__":
    main()
