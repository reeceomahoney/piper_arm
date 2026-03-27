"""Evaluate an advantage-conditioned policy across guidance scales.

Runs lerobot-eval for each guidance scale and prints a summary table.
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import draccus


@dataclass
class EvalGuidanceConfig:
    policy_path: str = "reece-omahoney/adv-libero-success-expert"
    guidance_scales: list[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5])


@draccus.wrap()
def main(cfg: EvalGuidanceConfig):
    results = {}

    for beta in cfg.guidance_scales:
        print(f"\n{'=' * 60}")
        print(f"  guidance_scale = {beta}")
        print(f"{'=' * 60}\n", flush=True)

        cmd = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_eval",
            "--config_path=configs/eval.yaml",
            f"--policy.path={cfg.policy_path}",
            f"--policy.guidance_scale={beta}",
        ]

        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"WARNING: eval failed for beta={beta}")
            continue

        # Find the most recent eval output
        eval_root = Path("outputs/eval")
        if not eval_root.exists():
            continue

        latest = max(eval_root.rglob("eval_info.json"), key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            info = json.load(f)

        overall = info.get("overall", {})
        pc_success = overall.get("pc_success", 0.0)
        results[beta] = pc_success

        print(f"\nguidance_scale={beta}: success={pc_success:.1f}%")

    # Summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Beta':>8} | {'Success Rate':>14}")
    print(f"{'-' * 8}-+-{'-' * 14}")
    for beta, pc in results.items():
        print(f"{beta:>8.1f} | {pc:>13.1f}%")


if __name__ == "__main__":
    main()
