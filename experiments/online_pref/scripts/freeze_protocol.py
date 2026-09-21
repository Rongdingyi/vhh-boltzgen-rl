#!/usr/bin/env python
"""Freeze the online_pref protocol (task book §49/§50)."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

import _common as C  # noqa: E402
from vhh_rl.online_pref import gates  # noqa: E402

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
    splits = {}
    for line in C.MANIFEST.open():
        row = json.loads(line)
        splits[row["case_id"]] = row.get("split")
    payload = {
        "version": 1,
        "method": "online_pref_spatiotemporal_dpo",
        "repo_commit": _git(C.ROOT),
        "boltzgen_commit": _git(Path("/share/home/rongdingyi/programs/proteingen/boltzgen")),
        "base_checkpoint": str(C.BASE_CKPT),
        "base_checkpoint_sha256": C.sha256(C.BASE_CKPT),
        "manifest_sha256": C.sha256(C.MANIFEST),
        "reward_scorer": {"name": "cdr_margin_guarded",
                          "cache": "runs/cf_opsd/reward_cache.sqlite"},
        "train_cases": C.TRAIN_CASES,
        "heldout8_cases": C.heldout8(),
        "k_siblings": C.K_SIBLINGS,
        "sampling_steps": C.SAMPLING_STEPS,
        "phase0_branch_progress": C.BRANCH_PROGRESS,
        "pair_policies": ["best_vs_worst", "best_vs_median"],
        "min_reward_gap": C.MIN_REWARD_GAP,
        "rounds": C.ROUNDS,
        "updates_per_round": C.UPDATES_PER_ROUND,
        "behavior_ema_decay": C.EMA_DECAY,
        "optimizer": {"beta": 10.0, "lr": 1.0e-5, "max_grad_norm": 1.0},
        "train_seeds": C.TRAIN_SEEDS,
        "eval_seed_offset": C.EVAL_SEED_OFFSET,
        "gate_thresholds": {
            "phase0": {"min_mean_delta": gates.PHASE0_MIN_MEAN_DELTA,
                       "min_seeds_positive": gates.PHASE0_MIN_SEEDS_POSITIVE,
                       "min_cases_ge": gates.PHASE0_MIN_CASES_GE,
                       "max_invalid": gates.MAX_INVALID},
            "phase1": {"min_suffix_full": gates.PHASE1_MIN_SUFFIX_FULL,
                       "min_suffix_prefix": gates.PHASE1_MIN_SUFFIX_PREFIX,
                       "max_prefix_full": gates.PHASE1_MAX_PREFIX_FULL},
            "phase2": {"min_suffix_full": gates.PHASE2_MIN_SUFFIX_FULL},
            "phase3": {"min_p80_p60": gates.PHASE3_MIN_P80_P60,
                       "min_eligible_ratio": gates.PHASE3_MIN_ELIGIBLE_RATIO},
            "phase4": {"min_minedit_rank": gates.PHASE4_MIN_MINEDIT_RANK},
            "final": {"min_final_offline": gates.FINAL_MIN_FINAL_OFFLINE,
                      "min_final_online": gates.FINAL_MIN_FINAL_ONLINE},
        },
        "phase0_arm_definitions": {
            "A": "offline fixed diff_only pairs, all changed positions, 100 updates",
            "B": "online sibling pairs, all changed positions, full sigma",
            "V": "same pair bank/schedule as B, verified (carrier) support control",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(json.dumps({"wrote": str(OUT), "sha256": C.sha256(OUT)}, indent=1))


if __name__ == "__main__":
    main()
