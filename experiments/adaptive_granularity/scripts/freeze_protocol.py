#!/usr/bin/env python
"""Phase A0: freeze provenance for AG-CF-DPO (task book §7)."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

import yaml

import _common as C  # noqa: E402

OUT = C.ROOT / "experiments/adaptive_granularity/configs/FROZEN_PROTOCOL.yaml"


def _git(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main() -> None:
    splits = C.manifest_splits()
    payload = {
        "version": 1,
        "method": "ag_cf_dpo",
        "repo_commit": _git(C.ROOT),
        "boltzgen_commit": _git(Path("/share/home/rongdingyi/programs/proteingen/boltzgen")),
        "base_checkpoint_sha256": C.sha256(C.BASE_CKPT),
        "pairs_train_sha256": C.sha256(C.PAIRS),
        "residue_credit_sha256": C.sha256(C.RESIDUE_CREDIT),
        "region_credit_sha256": C.sha256(C.REGION_CREDIT),
        "current_residue_weights_sha256": C.sha256(C.CURRENT_WEIGHTS),
        "train_cases": sorted(c for c, s in splits.items() if s == "train"),
        "pilot_train_cases": C.PILOT_TRAIN_CASES,
        "heldout4_cases": C.HELDOUT4,
        "heldout8_cases": C.heldout8(),
        "beta": 10.0,
        "lr": 1.0e-5,
        "eta_current": 0.75,
        "tol": 0.05,
        "rho_threshold": 0.70,
        "eval_seeds": "case.seed_base + 800000 (8 samples/case, 50 steps)",
        "pilot_updates": 100,
        "full_updates": 500,
        "train_seed": 20260913,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
