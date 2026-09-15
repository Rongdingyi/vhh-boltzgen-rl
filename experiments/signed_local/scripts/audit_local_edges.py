#!/usr/bin/env python
"""Validate the local edge dataset invariants (task book §15/§16/§83)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))
from vhh_rl.signed_local.edge_validator import validate_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=Path,
                        default=ROOT / "runs/signed_local/edges/train_edges.jsonl")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    report = validate_dataset(args.edges, split=args.split)
    out = ROOT / "runs/signed_local/edge_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
