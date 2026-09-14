#!/usr/bin/env python
"""Verify the paper-stage frozen protocol hashes (task book §55/§58 Step 0)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import yaml

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
PROTO = ROOT / "experiments/paper_stage/configs/FROZEN_PROTOCOL.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    proto = yaml.safe_load(PROTO.read_text())
    checks = [
        ("base_checkpoint", Path(proto["base_checkpoint"]), proto["base_checkpoint_sha256"]),
        ("pair_dataset", Path(proto["pair_dataset"]), proto["pair_dataset_sha256"]),
        ("valid100_manifest", Path(proto["data"]["valid100_manifest"]),
         proto["data"]["valid100_manifest_sha256"]),
        ("split_manifest", Path(proto["data"]["split_manifest"]),
         proto["data"]["split_manifest_sha256"]),
    ]
    scorer_root = Path(proto["scorer"]["predictor_root"])
    for name, expected in proto["scorer"]["ensemble_sha256"].items():
        checks.append((f"scorer/{name}", scorer_root / "models" / f"{name}.pt", expected))
    checks.append(("scorer/ood_reference",
                   scorer_root / "models" / "ensemble_ood_reference.json",
                   proto["scorer"]["ood_reference_sha256"]))
    failures = 0
    for name, path, expected in checks:
        if not path.is_file():
            print(f"[FAIL] {name}: missing {path}")
            failures += 1
            continue
        actual = sha256(path)
        ok = actual == expected
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {actual[:16]}...")
        failures += 0 if ok else 1
    print("PROTOCOL FROZEN" if failures == 0 else f"{failures} HASH MISMATCHES")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
