#!/usr/bin/env python
"""Append valid100 evaluation tasks (4 shards per checkpoint) to the paper-stage
task list, then submit the array job.

usage: add_valid100_tasks.py <arm_tag> <ckpt_path>
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
TASKS = ROOT / "runs/paper_stage/valid100_tasks.tsv"


def main() -> None:
    arm, ckpt = sys.argv[1], sys.argv[2]
    if not Path(ckpt).is_file():
        raise SystemExit(f"missing checkpoint {ckpt}")
    TASKS.parent.mkdir(parents=True, exist_ok=True)
    with TASKS.open("a") as fh:
        for shard in range(4):
            fh.write(f"{arm}\t{ckpt}\t{shard}\n")
    n = sum(1 for _ in TASKS.open())
    print(f"tasks now {n}; submit with: sbatch --array=0-{n - 1}%4 {ROOT}/experiments/paper_stage/scripts/sbatch_paper_valid100.sh")
    subprocess.run([
        "sbatch", f"--array=0-{n - 1}%4",
        str(ROOT / "experiments/paper_stage/scripts/sbatch_paper_valid100.sh"),
    ], check=True)


if __name__ == "__main__":
    main()
