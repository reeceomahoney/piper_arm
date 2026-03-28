"""Evaluate a policy multiple times and measure variance of success rates."""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import draccus
import numpy as np


@dataclass
class EvalVarianceConfig:
    policy_path: str = "reece-omahoney/adv-libero-base"
    n_runs: int = 5


@draccus.wrap()
def main(cfg: EvalVarianceConfig):
    success_rates: list[float] = []

    for i in range(cfg.n_runs):
        print(f"\n{'=' * 60}")
        print(f"  Run {i + 1}/{cfg.n_runs}")
        print(f"{'=' * 60}\n", flush=True)

        cmd = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_eval",
            "--config_path=configs/eval.yaml",
            f"--policy.path={cfg.policy_path}",
        ]

        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"WARNING: eval failed for run {i + 1}")
            continue

        eval_root = Path("outputs/eval")
        if not eval_root.exists():
            continue

        latest = max(eval_root.rglob("eval_info.json"), key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            info = json.load(f)

        pc_success = info.get("overall", {}).get("pc_success", 0.0)
        success_rates.append(pc_success)
        print(f"\nRun {i + 1}: success={pc_success:.1f}%")

    if not success_rates:
        print("No successful runs.")
        return

    rates = np.array(success_rates)
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Run':>6} | {'Success Rate':>14}")
    print(f"{'-' * 6}-+-{'-' * 14}")
    for i, rate in enumerate(success_rates):
        print(f"{i + 1:>6} | {rate:>13.1f}%")
    print(f"{'-' * 6}-+-{'-' * 14}")
    print(f"{'Mean':>6} | {rates.mean():>13.1f}%")
    print(f"{'Std':>6} | {rates.std():>13.1f}%")
    print(f"{'Min':>6} | {rates.min():>13.1f}%")
    print(f"{'Max':>6} | {rates.max():>13.1f}%")
    print(f"{'N':>6} | {len(success_rates):>14}")


if __name__ == "__main__":
    main()
