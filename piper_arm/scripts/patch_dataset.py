"""Download lerobot/libero, compute missing image stats, and re-upload to reeceomahoney/libero."""

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_stats
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_image_stats(
    dataset: LeRobotDataset, camera_keys: list[str], num_workers: int = 4
) -> dict:
    """Compute mean, std, min, max, and quantile stats for image keys."""
    loader = DataLoader(dataset, batch_size=64, num_workers=num_workers, shuffle=False)

    # Accumulators
    n = 0
    running_sum = {k: None for k in camera_keys}
    running_sum_sq = {k: None for k in camera_keys}
    running_min = {k: None for k in camera_keys}
    running_max = {k: None for k in camera_keys}

    for batch in tqdm(loader, desc="Computing image stats"):
        for key in camera_keys:
            # images are (B, C, H, W) float tensors
            imgs = batch[key].float()
            b = imgs.shape[0]
            # reduce over batch, height, width -> (C,)
            s = imgs.sum(dim=(0, 2, 3))
            s2 = (imgs**2).sum(dim=(0, 2, 3))
            pixels_per_img = imgs.shape[2] * imgs.shape[3]
            mn = imgs.amin(dim=(0, 2, 3))
            mx = imgs.amax(dim=(0, 2, 3))

            if running_sum[key] is None:
                running_sum[key] = s
                running_sum_sq[key] = s2
                running_min[key] = mn
                running_max[key] = mx
            else:
                running_sum[key] += s
                running_sum_sq[key] += s2
                running_min[key] = torch.minimum(running_min[key], mn)
                running_max[key] = torch.maximum(running_max[key], mx)
        n += b * pixels_per_img

    stats = {}
    for key in camera_keys:
        mean = running_sum[key] / n
        var = running_sum_sq[key] / n - mean**2
        std = var.clamp(min=0).sqrt()

        # Reshape to (C, 1, 1) to match lerobot convention
        mean_np = mean.reshape(-1, 1, 1).numpy()
        std_np = std.reshape(-1, 1, 1).numpy()
        min_np = running_min[key].reshape(-1, 1, 1).numpy()
        max_np = running_max[key].reshape(-1, 1, 1).numpy()

        stats[key] = {
            "mean": mean_np,
            "std": std_np,
            "min": min_np,
            "max": max_np,
        }

    return stats


def main():
    print("Loading dataset lerobot/libero...")
    dataset = LeRobotDataset("lerobot/libero", force_cache_sync=True)
    camera_keys = dataset.meta.camera_keys
    print(f"Camera keys: {camera_keys}")
    print(f"Current stats keys: {list(dataset.meta.stats.keys())}")

    missing_keys = [k for k in camera_keys if k not in dataset.meta.stats]
    if not missing_keys:
        print("All camera keys already have stats. Nothing to do.")
        return

    print(f"Missing stats for: {missing_keys}")
    print("Computing image stats...")
    image_stats = compute_image_stats(dataset, missing_keys)

    # Merge into existing stats
    for key, key_stats in image_stats.items():
        dataset.meta.stats[key] = {k: np.array(v) for k, v in key_stats.items()}

    print(f"Updated stats keys: {list(dataset.meta.stats.keys())}")

    # Write updated stats locally
    write_stats(dataset.meta.stats, dataset.root)
    print("Wrote updated stats.json")

    # Re-upload to reeceomahoney/libero
    dataset.repo_id = "reece-omahoney/libero"
    print(f"Pushing to {dataset.repo_id}...")
    dataset.push_to_hub(upload_large_folder=True)
    print("Done!")


if __name__ == "__main__":
    main()
