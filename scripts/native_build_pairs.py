#!/usr/bin/env python
"""Build rank-symmetric preference pairs from the native pool (task book §20-22)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.native_atom14.preference_dataset import build_pairs, gap_stats  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-root", type=Path, default=ROOT / "runs/native_pool")
    parser.add_argument("--split", default="train")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or (args.pool_root / f"pairs_{args.split}.jsonl")
    pairs = build_pairs(args.pool_root, args.split, top_k=args.top_k, out_path=out)
    stats = gap_stats(pairs)
    print(json.dumps(stats, indent=1))
    (args.pool_root / f"pairs_{args.split}_gap_stats.json").write_text(
        json.dumps(stats, indent=1))
    print(f"wrote {len(pairs)} pairs -> {out}")


if __name__ == "__main__":
    main()
