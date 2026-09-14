#!/usr/bin/env python
"""Phase A1/A2: build single-residue and region counterfactual sequences (§9/§17)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.credit.counterfactual import (  # noqa: E402
    build_region_cfs, build_single_residue_cfs, sha1,
)
from vhh_rl.credit.cdr_regions import region_map  # noqa: E402
from vhh_rl.data.case import RLCase  # noqa: E402

POOL = ROOT / "runs/native_pool"
OUT = ROOT / "runs/next_stage/counterfactual"


def load_cases() -> dict[str, RLCase]:
    cases = {}
    for line in (ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl").open():
        row = json.loads(line)
        cases[row["case_id"]] = RLCase(
            case_id=row["case_id"], structure_path=Path(row["structure_path"]),
            chain_id=row.get("chain_id", "A"), full_sequence=row["full_sequence"],
            design_positions=tuple(row["design_positions"]),
            fr_positions=tuple(row["fr_positions"]), split=row.get("split", "train"),
            seed_base=int(row.get("seed_base", 0)),
        )
    return cases


def load_sequences() -> dict[str, str]:
    seqs = {}
    for meta in (POOL / "train").glob("*/metadata.jsonl"):
        for line in meta.open():
            r = json.loads(line)
            seqs[r["sample_id"]] = r["decoded_sequence"]
    return seqs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    pairs = [json.loads(l) for l in (POOL / "pairs_train.jsonl").open()]
    seqs = load_sequences()
    design = {cid: cases[cid].design_positions for cid in cases}
    fixed = {cid: cases[cid].fr_positions for cid in cases}
    regions = {cid: region_map(cases[cid]) for cid in cases}

    # sanity: design == cdr1+cdr2+cdr3 for every case
    for cid, c in cases.items():
        rm = regions[cid]
        assert sorted(rm["cdr1"] + rm["cdr2"] + rm["cdr3"]) == sorted(c.design_positions), cid

    rows = []
    n_single = n_region = 0
    for pair in pairs:
        cid = pair["case_id"]
        pid = f"{pair['case_id']}:{pair['winner_sample_id']}:{pair['loser_sample_id']}"
        rows.append({"cf_id": f"{pid}:orig_w", "pair_id": pid, "case_id": cid,
                     "kind": "original_winner", "position": None, "region": None,
                     "sequence": seqs[pair["winner_sample_id"]],
                     "sequence_sha1": sha1(seqs[pair["winner_sample_id"]])})
        rows.append({"cf_id": f"{pid}:orig_l", "pair_id": pid, "case_id": cid,
                     "kind": "original_loser", "position": None, "region": None,
                     "sequence": seqs[pair["loser_sample_id"]],
                     "sequence_sha1": sha1(seqs[pair["loser_sample_id"]])})
        singles = build_single_residue_cfs(pair, seqs, design, fixed)
        for s in singles:
            s["cf_id"] = f"{pid}:{s['kind']}:{s['position']}"
            rows.append(s)
            n_single += 1
        regs = build_region_cfs(pair, seqs, regions[cid], fixed)
        for r in regs:
            r["cf_id"] = f"{pid}:{r['kind']}:{r['region']}"
            rows.append(r)
            n_region += 1
    # dedupe by cf sequence sha1 (same residue pair can repeat across pairs? keep unique)
    with (OUT / "counterfactual_sequences.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    n_unique = len({r["sequence_sha1"] for r in rows})
    summary = {"n_rows": len(rows), "n_unique_sequences": n_unique,
               "n_single_residue": n_single, "n_region": n_region,
               "n_originals": 2 * len(pairs), "n_pairs": len(pairs)}
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
