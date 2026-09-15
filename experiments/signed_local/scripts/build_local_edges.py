#!/usr/bin/env python
"""Build the SL-CF-DPO train correction edge dataset (task book §17-§19)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))
from vhh_rl.data.case import RLCase  # noqa: E402
from vhh_rl.signed_local.edge_builder import (  # noqa: E402
    build_edges, load_cf_rewards, load_credit,
)

OUT = ROOT / "runs/signed_local/edges"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="*", default=["both_negative", "sign_flip"])
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()
    cases = {}
    for line in (ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl").open():
        row = json.loads(line)
        if row.get("split") != "train":
            continue
        cases[row["case_id"]] = RLCase(
            case_id=row["case_id"], structure_path=Path(row["structure_path"]),
            chain_id=row.get("chain_id", "A"), full_sequence=row["full_sequence"],
            design_positions=tuple(row["design_positions"]),
            fr_positions=tuple(row["fr_positions"]), split="train",
            seed_base=int(row.get("seed_base", 0)))
    pairs = [json.loads(l) for l in (ROOT / "runs/native_pool/pairs_train.jsonl").open()]
    summary = build_edges(cases, pairs, load_credit(), load_cf_rewards(),
                          out_dir=OUT, split="train", classes=set(args.classes),
                          max_pairs=args.max_pairs)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
