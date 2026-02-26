"""Rerun visualization for rollout traces captured by eval_trace.

Loads saved MP4 videos and NPZ distance traces, then displays them in the
Rerun viewer with synced timeline (step number).

Usage:
    # Visualize all episodes in a trace directory:
    python -m piper_arm.visualize \
        --trace-dir outputs/eval_trace/latest

    # Visualize a specific episode:
    python -m piper_arm.visualize \
        --trace-dir outputs/eval_trace/latest \
        --episode 0
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import draccus
import numpy as np
import rerun as rr


def _load_video_frames(path: Path) -> list[np.ndarray]:
    """Load all frames from an MP4 as a list of RGB uint8 arrays."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def visualize_episode(
    trace_dir: Path,
    episode: int,
    task_desc: str = "",
) -> None:
    """Log a single episode's data to the active Rerun recording."""
    agentview_path = trace_dir / f"episode_{episode}_agentview.mp4"
    eye_in_hand_path = trace_dir / f"episode_{episode}_eye_in_hand.mp4"
    trace_path = trace_dir / f"episode_{episode}_trace.npz"

    if not trace_path.exists():
        print(f"Trace not found: {trace_path}")
        return

    # Load trace
    trace_data = np.load(trace_path)
    trace_steps = trace_data["steps"]
    trace_distances = trace_data["distances"]

    # Load video frames
    agentview_frames = (
        _load_video_frames(agentview_path) if agentview_path.exists() else []
    )
    eye_in_hand_frames = (
        _load_video_frames(eye_in_hand_path) if eye_in_hand_path.exists() else []
    )

    n_video_frames = max(len(agentview_frames), len(eye_in_hand_frames))
    if n_video_frames == 0 and len(trace_steps) == 0:
        print(f"No data for episode {episode}")
        return

    label = f"Episode {episode}"
    if task_desc:
        label += f": {task_desc}"
    print(
        f"Logging {label} ({n_video_frames} frames, "
        f"{len(trace_steps)} distance samples)"
    )

    # Log video frames at each step
    for step in range(n_video_frames):
        rr.set_time("step", sequence=step)

        if step < len(agentview_frames):
            rr.log("cameras/agentview", rr.Image(agentview_frames[step]))

        if step < len(eye_in_hand_frames):
            rr.log("cameras/eye_in_hand", rr.Image(eye_in_hand_frames[step]))

    # Log distance trace at forward-pass steps
    for step, dist in zip(trace_steps, trace_distances):
        rr.set_time("step", sequence=int(step))
        rr.log("metrics/mahalanobis_distance", rr.Scalars(float(dist)))


@dataclass
class VisualizeConfig:
    trace_dir: str = "outputs/eval_trace/latest"
    episode: Optional[int] = None
    port: int = 9876


@draccus.wrap()
def main(cfg: VisualizeConfig):
    trace_dir = Path(cfg.trace_dir).resolve()

    if not trace_dir.exists():
        print(f"Trace directory not found: {trace_dir}")
        return

    # Load summary for task descriptions
    summary_path = trace_dir / "summary.json"
    episode_meta: dict[int, dict] = {}
    if summary_path.exists():
        with open(summary_path) as f:
            for entry in json.load(f):
                episode_meta[entry["episode"]] = entry

    # Determine which episodes to visualize
    if cfg.episode is not None:
        episodes = [cfg.episode]
    else:
        # Find all episode trace files
        trace_files = sorted(trace_dir.glob("episode_*_trace.npz"))
        episodes = []
        for tf in trace_files:
            # Parse episode number from filename: episode_0_trace.npz
            parts = tf.stem.split("_")
            episodes.append(int(parts[1]))

    if not episodes:
        print("No episodes found")
        return

    # Spawn the viewer once, then send each episode as a separate
    # recording so they appear in the recording selector.
    rr.init("eval_trace", spawn=True)
    addr = f"rerun+http://127.0.0.1:{cfg.port}/proxy"

    for ep in episodes:
        meta = episode_meta.get(ep, {})
        task_desc = meta.get("task_description", "")
        rec = rr.new_recording(application_id=f"episode_{ep}")
        rec.connect_grpc(addr)
        with rec:
            visualize_episode(trace_dir, ep, task_desc=task_desc)

    print("Rerun viewer launched. Close the viewer window to exit.")


if __name__ == "__main__":
    main()
