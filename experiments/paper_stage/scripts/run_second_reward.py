#!/usr/bin/env python
"""P1-B driver: submit second-reward (ESM-C CDR PLL) uniform/CF trainings and evals.

usage: run_second_reward.py train | eval | all
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
SEC = ROOT / "runs/paper_stage/second_reward"


def submit(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def train() -> None:
    assert (SEC / "weights.json").is_file(), "run build_second_reward_credit.py first"
    pairs = str(SEC / "pairs_train.jsonl")
    submit(["sbatch", str(ROOT / "scripts/sbatch_next_cf_dpo.sh"),
            "--variant", "uniform_all", "--pairs", pairs,
            "--max-steps", "500", "--checkpoint-every", "50",
            "--output-dir", str(SEC / "uniform"), "--tag", "sr_uniform"])
    submit(["sbatch", str(ROOT / "scripts/sbatch_next_cf_dpo.sh"),
            "--variant", "cf", "--weights", str(SEC / "weights.json"),
            "--pairs", pairs, "--max-steps", "500", "--checkpoint-every", "50",
            "--output-dir", str(SEC / "cf"), "--tag", "sr_cf"])


def eval_() -> None:
    script = str(ROOT / "experiments/paper_stage/scripts/sbatch_second_reward_eval.sh")
    submit(["sbatch", "--export=ALL,TAG=base", script])
    for tag, run in (("sr_uniform", "uniform"), ("sr_cf", "cf")):
        ckpt = SEC / run / "checkpoint_0500.pt"
        submit(["sbatch", f"--export=ALL,TAG={tag},CKPT={ckpt}", script])


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("train", "all"):
        train()
    if stage in ("eval", "all"):
        eval_()


if __name__ == "__main__":
    main()
