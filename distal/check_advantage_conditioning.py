"""Check whether a trained advantage policy responds to advantage tokens.

Loads the policy and dataset, then compares flow-matching loss outputs
with positive, negative, and zeroed advantage labels on the same batch
using identical noise and timesteps. If the model ignores advantage
conditioning, the deltas will be near zero.
"""

from dataclasses import dataclass
from typing import cast as typecast

import draccus
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot_policy_advantage.configuration_advantage import AdvantageConfig
from lerobot_policy_advantage.modeling_advantage import (
    AdvantagePolicy,
)
from torch.utils.data import DataLoader

import lerobot_policy_advantage as lerobot_policy_advantage


@dataclass
class CheckAdvantageConfig:
    policy_path: str = "reece-omahoney/adv-libero-base"
    dataset_repo_id: str = "lerobot/libero"
    batch_size: int = 32
    num_batches: int = 10
    num_workers: int = 4
    device: str = "cuda"


def forward_with_label(policy, batch, label_value, noise, time):
    """Run policy forward with a fixed advantage label and shared noise/time.

    Returns per-sample losses (B,).
    """
    batch = dict(batch)
    bsize = batch["observation.language.tokens"].shape[0]
    batch["observation.language.advantage_label"] = torch.full(
        (bsize, 1), label_value, dtype=torch.long, device=batch["action"].device
    )
    loss, _ = policy.forward(batch, noise=noise, time=time, reduction="none")
    return loss


@draccus.wrap()
def main(cfg: CheckAdvantageConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Load policy
    print("Loading policy...")
    policy_cfg = typecast(
        AdvantageConfig, PreTrainedConfig.from_pretrained(cfg.policy_path)
    )
    policy_cfg.pretrained_path = cfg.policy_path  # type: ignore[assignment]
    policy_cfg.device = str(device)
    policy_cfg.fixed_advantage = False
    policy = AdvantagePolicy(policy_cfg)
    policy.to(device)
    policy.eval()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg, pretrained_path=cfg.policy_path
    )

    # Load dataset with action chunking matching the policy's chunk_size
    print("Loading dataset...")
    fps = 10
    chunk_size = policy_cfg.chunk_size
    delta_timestamps = {"action": [i / fps for i in range(chunk_size)]}
    dataset = LeRobotDataset(cfg.dataset_repo_id, delta_timestamps=delta_timestamps)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    bsize = cfg.batch_size

    # Accumulators
    delta_pos_neg_sum = 0.0
    mean_loss_sum = 0.0
    batches_run = 0

    print(f"\nRunning {cfg.num_batches} batches of {cfg.batch_size} samples...\n")

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= cfg.num_batches:
            break

        batch = preprocessor(batch)

        # Fix noise and timestep for fair comparison across conditions
        actions = policy.prepare_action(batch)
        noise = policy.model.sample_noise(actions.shape, device)
        time = policy.model.sample_time(bsize, device)

        with torch.no_grad():
            lp = forward_with_label(policy, batch, 1, noise, time)
            ln = forward_with_label(policy, batch, 0, noise, time)

        delta_pn = (lp - ln).abs().mean().item()
        mean_loss = lp.mean().item()

        delta_pos_neg_sum += delta_pn
        mean_loss_sum += mean_loss
        batches_run += 1

        print(
            f"  batch {batch_idx + 1:3d}  "
            f"|pos-neg|={delta_pn:.6f}  "
            f"mean_loss={mean_loss:.6f}"
        )

    if batches_run == 0:
        print("No batches were run.")
        return

    # Summary
    avg_pn = delta_pos_neg_sum / batches_run
    avg_loss = mean_loss_sum / batches_run

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Mean loss (positive adv):   {avg_loss:.6f}")
    print(f"  Mean |positive - negative|: {avg_pn:.6f}")
    print()

    ratio_pn = avg_pn / avg_loss if avg_loss > 0 else float("inf")
    print(f"  |pos-neg| / loss ratio:     {ratio_pn:.4f}")
    print()

    if ratio_pn < 0.01:
        print(
            "  DIAGNOSIS: Advantage conditioning has NEGLIGIBLE effect (<1% of loss)."
        )
        print("  The model is likely ignoring the advantage tokens entirely.")
    elif ratio_pn < 0.05:
        print("  DIAGNOSIS: Advantage conditioning has WEAK effect (1-5% of loss).")
        print("  The model may be partially using advantage tokens.")
    else:
        print(
            f"  DIAGNOSIS: Advantage conditioning has MEANINGFUL effect "
            f"({ratio_pn * 100:.1f}% of loss)."
        )
        print("  The model is responding to advantage token changes.")


if __name__ == "__main__":
    main()
