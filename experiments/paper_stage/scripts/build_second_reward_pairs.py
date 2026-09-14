#!/usr/bin/env python
"""Second-reward pipeline 1/3: score the base train pool with the second reward
and build rank-symmetric preference pairs (same protocol as round-1 pairs).

Writes runs/paper_stage/second_reward/{base_pool_scored.jsonl,pairs_train.jsonl,
pairs_train_gap_stats.json}
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
OUT = ROOT / "runs/paper_stage/second_reward"
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    from esmc_reward import ESMCReward, load_cases

    OUT.mkdir(parents=True, exist_ok=True)
    cases = load_cases("train")
    samples = []
    for meta_path in sorted((ROOT / "runs/native_pool/train").glob("*/metadata.jsonl")):
        for line in meta_path.open():
            r = json.loads(line)
            if r["reward_raw"] is None or r["contains_UNK"] or r["FR_mismatch_count"] > 0:
                continue
            samples.append(r)
    print(f"{len(samples)} base-pool samples")
    scorer = ESMCReward(cache_path=OUT / "esmc_cache.sqlite")
    records = [{"key": s["sample_id"], "case_id": s["case_id"], "sequence": s["decoded_sequence"]}
               for s in samples]
    scored = scorer.score(records, cases, batch_size=32)
    by_id = {r["key"]: r for r in scored}
    with (OUT / "base_pool_scored.jsonl").open("w") as fh:
        for s in samples:
            r = by_id[s["sample_id"]]
            fh.write(json.dumps({
                "sample_id": s["sample_id"], "case_id": s["case_id"],
                "sequence": s["decoded_sequence"],
                "cdr_pll": r.get("cdr_pll"), "full_pll": r.get("full_pll"),
                "status": r.get("status"),
            }) + "\n")

    # rank-symmetric pairs per case: top-8 vs bottom-8 (by cdr_pll), w > l
    per_case: dict[str, list[dict]] = {}
    for s in samples:
        r = by_id[s["sample_id"]]
        if r.get("status") != "PASS":
            continue
        per_case.setdefault(s["case_id"], []).append(
            {"sample_id": s["sample_id"], "reward": float(r["cdr_pll"])})
    pairs = []
    for cid, rows in sorted(per_case.items()):
        if len(rows) < 16:
            continue
        ranked = sorted(rows, key=lambda r: r["reward"], reverse=True)
        for w, l in zip(ranked[:8], ranked[-8:][::-1]):
            if w["reward"] <= l["reward"]:
                continue
            pairs.append({
                "case_id": cid,
                "winner_sample_id": w["sample_id"], "loser_sample_id": l["sample_id"],
                "winner_reward": w["reward"], "loser_reward": l["reward"],
                "reward_gap": w["reward"] - l["reward"],
                "condition_id": cid, "design_mask_id": cid,
                "reward_name": "esmc_vhh_cdr_pll",
            })
    with (OUT / "pairs_train.jsonl").open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    gaps = [p["reward_gap"] for p in pairs]
    stats = {"n_pairs": len(pairs), "n_cases": len({p["case_id"] for p in pairs}),
             "gap_median": st.median(gaps) if gaps else None,
             "gap_mean": st.mean(gaps) if gaps else None}
    (OUT / "pairs_train_gap_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
