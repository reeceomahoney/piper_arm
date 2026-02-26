"""Rollout trace capture — run episodes with no interventions, save camera
videos and Mahalanobis distance traces for offline visualization.

Reuses Phase 1 (Gaussian fitting) and core rollout logic from eval_dist,
but strips all intervention machinery. Outputs per-episode MP4 videos,
NPZ distance traces, and a summary JSON.

Usage:
    python -m piper_arm.eval_trace \
        --policy-path lerobot/pi05_libero_finetuned \
        --dataset reece-omahoney/libero \
        --n-episodes 1

    # With cached Gaussian stats:
    python -m piper_arm.eval_trace \
        --policy-path lerobot/pi05_libero_finetuned \
        --load-stats outputs/eval_dist/.../gauss_stats.npz \
        --n-episodes 3
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import draccus
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import inside_slurm
from tqdm import tqdm

from piper_arm.eval_dist import (
    _get_action_queue,
    compute_mahalanobis_np,
    embed_prefix_pooled,
    fit_gaussian_from_dataset,
)

# Camera keys in observation["pixels"] → short names for output files
CAMERA_KEYS = {
    "image": "agentview",
    "image2": "eye_in_hand",
}


def _write_video(frames: list[np.ndarray], path: Path, fps: int = 10) -> None:
    """Write a list of uint8 HWC frames to an MP4 file."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        # observation frames are RGB, VideoWriter expects BGR
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _run_episode_capture(
    policy: Union[PI05Policy, SmolVLAPolicy],
    vec_env: Any,
    preprocessor: Any,
    postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    gauss_mean: np.ndarray,
    gauss_cov_inv: np.ndarray,
    seed: int,
    desc: str = "",
) -> dict[str, Any]:
    """Run a single episode, capturing camera frames and Mahalanobis traces.

    No interventions — the episode runs to completion or truncation.

    Returns:
        Dict with keys: success, n_steps, mean_distance,
        camera_frames (dict of lists), trace_steps, trace_distances.
    """
    max_steps = vec_env.call("_max_episode_steps")[0]
    observation, info = vec_env.reset(seed=[seed])
    policy.reset()

    success = False
    done = False
    step = 0
    trace_steps: list[int] = []
    trace_distances: list[float] = []
    dist_history: list[float] = []
    camera_frames: dict[str, list[np.ndarray]] = {
        name: [] for name in CAMERA_KEYS.values()
    }

    for step in tqdm(
        range(max_steps),
        desc=desc,
        leave=False,
        disable=inside_slurm(),
    ):
        if done:
            break

        # Capture raw pixel frames before preprocessing
        pixels = observation["pixels"]
        for obs_key, name in CAMERA_KEYS.items():
            if obs_key in pixels:
                # Shape: (1, H, W, 3) uint8 — take the first env
                frame = pixels[obs_key][0]
                camera_frames[name].append(frame)

        observation = preprocess_observation(observation)
        observation = add_envs_task(vec_env, observation)
        observation = env_preprocessor(observation)
        observation = preprocessor(observation)

        with torch.inference_mode():
            policy.eval()
            action_queue = _get_action_queue(policy)

            if len(action_queue) == 0:
                emb = embed_prefix_pooled(policy, observation)
                emb_np = emb.cpu().numpy()
                dist = compute_mahalanobis_np(emb_np, gauss_mean, gauss_cov_inv)

                dist_val = dist[0].item()
                dist_history.append(dist_val)
                trace_steps.append(step)
                trace_distances.append(dist_val)
                _ = True  # forward pass happened (distance already recorded)

                action_chunk = policy.predict_action_chunk(observation)
                n_action_steps = policy.config.n_action_steps
                action_chunk = action_chunk[:, :n_action_steps]
                action_queue.extend(action_chunk.transpose(0, 1))

            action = action_queue.popleft()

        action = postprocessor(action)
        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action_np = action_transition[ACTION].to("cpu").numpy()

        observation, _, terminated, truncated, info = vec_env.step(action_np)

        if "final_info" in info:
            if info["final_info"]["is_success"].item():
                success = True

        done = bool(terminated | truncated)

    mean_dist = float(np.mean(trace_distances)) if trace_distances else float("nan")

    return {
        "success": success,
        "n_steps": step + 1 if max_steps > 0 else 0,
        "mean_distance": mean_dist,
        "camera_frames": camera_frames,
        "trace_steps": trace_steps,
        "trace_distances": trace_distances,
    }


@dataclass
class EvalTraceConfig:
    policy_path: str = "reece-omahoney/smolvla-libero-16-chunk"
    dataset: str = "reece-omahoney/libero"
    n_episodes: int = 1
    batch_size: int = 32
    num_workers: int = 8
    load_stats: Optional[str] = os.path.join(
        os.environ.get("OUTPUT_DIR", "outputs"),
        "eval_dist/2026-02-24/17-07-42/gauss_stats.npz",
    )
    output_dir: str = os.path.join(
        os.environ.get("OUTPUT_DIR", "outputs"), "eval_trace"
    )


@draccus.wrap()
def main(cfg: EvalTraceConfig):
    # ── Load policy ──
    suite_name = "libero_10"
    env_cfg = LiberoEnvConfig(suite_name, fps=10, task_ids=[9])
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    policy_cfg.pretrained_path = Path(cfg.policy_path)

    envs = make_env(env_cfg, n_envs=1)

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    assert isinstance(policy, (PI05Policy, SmolVLAPolicy))
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=str(policy_cfg.pretrained_path)
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg, policy_cfg
    )

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
            dataset=cfg.dataset,
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

    # ── Phase 2: Rollout with capture ──
    summary_entries: list[dict[str, Any]] = []

    for task_id, vec_env in envs[suite_name].items():
        task_desc = vec_env.call("task_description")[0]
        n_tasks = len(envs[suite_name])
        print(f"\n=== Task {task_id + 1}/{n_tasks}: {task_desc} ===")

        for ep in range(cfg.n_episodes):
            result = _run_episode_capture(
                policy=policy,
                vec_env=vec_env,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                gauss_mean=gauss_mean,
                gauss_cov_inv=gauss_cov_inv,
                seed=ep,
                desc=f"  Ep {ep}",
            )

            # Save videos
            for name in CAMERA_KEYS.values():
                video_path = output_dir / f"episode_{ep}_{name}.mp4"
                _write_video(result["camera_frames"][name], video_path)

            # Save trace
            trace_path = output_dir / f"episode_{ep}_trace.npz"
            np.savez(
                trace_path,
                steps=np.array(result["trace_steps"]),
                distances=np.array(result["trace_distances"]),
            )

            status = "OK" if result["success"] else "FAIL"
            print(
                f"  Episode {ep}: {status} | "
                f"{result['n_steps']} steps | "
                f"mean_dist={result['mean_distance']:.2f}"
            )

            summary_entries.append(
                {
                    "episode": ep,
                    "task_id": task_id,
                    "task_description": task_desc,
                    "success": result["success"],
                    "n_steps": result["n_steps"],
                    "mean_distance": result["mean_distance"],
                }
            )

        vec_env.close()

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary_entries, f, indent=2)

    print(f"\nOutputs saved to {output_dir}")


if __name__ == "__main__":
    main()
