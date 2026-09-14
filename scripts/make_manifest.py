#!/usr/bin/env python3
"""Build the RL manifest jsonl from the guidance project's valid100 manifest.

Fields per case (task book §8): case_id, structure_path (canonical backbone),
chain_id, full_sequence, design_positions, fr_positions, split, seed_base,
design_mask_source.  Splits: the guidance project has no clean RL train split
on these cases -- every case was scored/analysed before.  We therefore mark
them split=train_debug (the §9 rule writes NO CLEAN TRAIN SPLIT FOUND into the
audit and limits round 1 to debug/overfit unless a fresh split is built).

Usage:
  python scripts/make_manifest.py --manifest <guidance valid100.csv> \
      --pool-root <g0 pool root> --out runs/round1/rl_manifest.jsonl [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

GUIDANCE = Path("/share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance")
sys.path.insert(0, str(GUIDANCE / "scripts/cdr_all"))

from common_cdr_all import load_cases  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--pool-root", default=None)
    args = parser.parse_args()

    cases = load_cases(
        None if args.limit == 0 else None,
        manifest=Path(args.manifest) if args.manifest else None,
        pool_root=Path(args.pool_root) if args.pool_root else None,
    )
    cases = cases[: args.limit] if args.limit else cases
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for case in cases:
            handle.write(
                json.dumps(
                    {
                        "case_id": case.case_id,
                        "structure_path": str(case.backbone_path),
                        "chain_id": "A",
                        "full_sequence": case.native_sequence,
                        "design_positions": list(case.design_positions),
                        "fr_positions": list(case.fixed_positions),
                        "split": "train_debug",
                        "seed_base": case.seed_base,
                        "design_mask_source": "vhh_esmc_guidance valid100 manifest (project CDR mask; task book §8)",
                    }
                )
                + "\n"
            )
    print(f"wrote {len(cases)} cases -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
