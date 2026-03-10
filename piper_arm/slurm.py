"""Utility for submitting jobs to SLURM cluster."""

import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import draccus
from fabric import Connection

REMOTE_HOST = "htc"
REMOTE_PATH = "/data/engs-robotics-ml/kebl6123/piper_arm"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SlurmConfig:
    command: str = "make train-advantage"

    # SLURM options
    time: int = 3
    gpu: str = "l40s"
    ngpu: int = 1
    cpus: int = 16
    mem: str = "8G"

    # Debug
    dry_run: bool = False


def build_sbatch_script(cfg: SlurmConfig) -> str:
    sbatch_opts = {
        "nodes": 1,
        "ntasks-per-node": cfg.cpus,
        "mem-per-cpu": cfg.mem,
        "time": f"{cfg.time}:00:00",
        "partition": "short",
        "gres": f"gpu:{cfg.gpu}:{cfg.ngpu}",
        "output": "slurm/slurm-%j.out",
    }
    header = "\n".join(f"#SBATCH --{k}={v}" for k, v in sbatch_opts.items())

    body = " ".join(
        [
            "singularity run --nv",
            '--env "WANDB_API_KEY=${WANDB_API_KEY}"',
            '--env "HF_TOKEN=${HF_TOKEN}"',
            f'--env "PYTHONPATH={REMOTE_PATH}:{REMOTE_PATH}/lerobot_policy_advantage"',
            f"container.sif {cfg.command}",
        ]
    )

    return f"#!/bin/bash\n{header}\n\nset -euo pipefail\n{body}\n"


def sync() -> None:
    subprocess.run(
        [
            "rsync",
            "-avz",
            "--filter=:- .gitignore",
            f"{PROJECT_ROOT}/",
            f"{REMOTE_HOST}:{REMOTE_PATH}",
        ],
        check=True,
    )


@draccus.wrap()
def main(cfg: SlurmConfig):
    script = build_sbatch_script(cfg)

    if cfg.dry_run:
        print("=== sbatch script ===")
        print(script)
        return

    print("Syncing...")
    sync()

    print("Submitting...")
    conn = Connection(REMOTE_HOST)
    conn.put(StringIO(script), f"{REMOTE_PATH}/submit.sh")
    conn.run(f"cd {REMOTE_PATH} && mkdir -p slurm && sbatch submit.sh")


if __name__ == "__main__":
    main()
