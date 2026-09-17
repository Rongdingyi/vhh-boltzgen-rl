#!/usr/bin/env python
"""Gate 0: freeze the branch_distill protocol (task book §20/§61)."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

import _common as C  # noqa: E402
from vhh_rl.branch_distill import gates  # noqa: E402

OUT = C.FROZEN


def _git(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if OUT.is_file() and not args.force:
        raise SystemExit(f"{OUT} exists; refusing to overwrite without --force")
    payload = {
        "version": 1,
        "method": "branch_distill",
        "repo_commit": _git(C.ROOT),
        "boltzgen_commit": _git(Path("/share/home/rongdingyi/programs/proteingen/boltzgen")),
        "base_checkpoint": str(C.BASE_CKPT),
        "base_checkpoint_sha256": C.sha256(C.BASE_CKPT),
        "reward_scorer": {
            "module": "vhh_rl.native_atom14.reward.make_reward_adapter",
            "cache": "runs/cf_opsd/reward_cache.sqlite",
            "name": "cdr_margin_guarded",
        },
        "train_cases": C.TRAIN_CASES,
        "heldout_cases": C.HELDOUT_CASES,
        "k_siblings": C.K_SIBLINGS,
        "sampling_steps": C.SAMPLING_STEPS,
        "candidate_progress": C.CANDIDATE_PROGRESS,
        "eval_seed_offset": C.EVAL_SEED_OFFSET,
        "gate_thresholds": {
            "gate1_group": {"valid_rate": gates.GATE1_MIN_VALID_RATE,
                            "unique": gates.GATE1_MIN_UNIQUE,
                            "reward_std": gates.GATE1_MIN_REWARD_STD,
                            "best_minus_median": gates.GATE1_MIN_BEST_MINUS_MEDIAN,
                            "teacher_vs_median_hamming": gates.GATE1_MIN_TEACHER_HAMMING},
            "gate1_progress": {"min_groups_pass": gates.GATE1_MIN_GROUPS_PASS,
                               "groups": gates.GATE1_GROUPS_PER_PROGRESS},
            "gate2": {"replay_abs": gates.GATE2_MIN_REPLAY_ABS,
                      "mse_ratio": gates.GATE2_MSE_RATIO,
                      "median_mse_ratio": gates.GATE2_MEDIAN_MSE_RATIO,
                      "min_improved": gates.GATE2_MIN_IMPROVED,
                      "min_downstream": gates.GATE2_MIN_DOWNSTREAM,
                      "safety_slack": gates.GATE2_SAFETY_SLACK},
            "gate3": {"delta_D_A": gates.GATE3_MIN_D_A,
                      "cases_D_ge_A": gates.GATE3_MIN_CASES_D_GE_A,
                      "delta_D_B": gates.GATE3_MIN_D_B,
                      "delta_D_C": gates.GATE3_MIN_D_C,
                      "max_invalid": gates.GATE3_MAX_INVALID},
        },
        "min_reward_gap": 0.30,
        "optimizer": {"lr": 1.0e-5, "weight_decay": 0.0, "max_grad_norm": 1.0,
                      "updates_gate2": 40, "updates_gate3_total": 100,
                      "rounds": 4, "updates_per_round": 25},
        "behavior_ema_decay": 0.99,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(json.dumps({"wrote": str(OUT), "sha256": C.sha256(OUT)}, indent=1))


if __name__ == "__main__":
    main()
