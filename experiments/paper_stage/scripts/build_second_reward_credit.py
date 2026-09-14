#!/usr/bin/env python
"""Second-reward pipeline 2/3: CF sequences + credit + weights for the second
reward (task book §41, same math as the main CF-DPO).

Reads  runs/paper_stage/second_reward/pairs_train.jsonl + base_pool_scored.jsonl
Writes counterfactual_credit.csv, weights.json (variant key "cf"), credit_stats.json
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
OUT = ROOT / "runs/paper_stage/second_reward"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vhh_rl.credit.controls import cf_weights  # noqa: E402
from vhh_rl.credit.counterfactual import build_single_residue_cfs  # noqa: E402


def main() -> None:
    from esmc_reward import ESMCReward, load_cases

    pairs = [json.loads(l) for l in (OUT / "pairs_train.jsonl").open()]
    base = {json.loads(l)["sample_id"]: json.loads(l)
            for l in (OUT / "base_pool_scored.jsonl").open()}
    cases = load_cases("train")
    # sequences per sample id
    seq_of = {s["sample_id"]: s["sequence"] for s in base.values()}
    design = {cid: tuple(c["design_positions"]) for cid, c in cases.items()}
    fixed = {}
    for cid, c in cases.items():
        design_set = set(c["design_positions"])
        fixed[cid] = tuple(i for i in range(len(c["native_sequence"])) if i not in design_set)

    # CF sequences for the new pairs
    cf_rows = []
    for pair in pairs:
        for row in build_single_residue_cfs(pair, seq_of, design, fixed):
            row["cf_id"] = f"{row['pair_id']}:{row['kind']}:{row['position']}"
            row["reward_name"] = "esmc_vhh_cdr_pll"
            cf_rows.append(row)
    print(f"{len(cf_rows)} single-residue CF sequences")
    scorer = ESMCReward(cache_path=OUT / "esmc_cache.sqlite")
    records = [{"key": r["cf_id"], "case_id": r["case_id"], "sequence": r["sequence"]}
               for r in cf_rows]
    scored = scorer.score(records, cases, batch_size=32)
    with (OUT / "counterfactual_scores.jsonl").open("w") as fh:
        for r in scored:
            fh.write(json.dumps(r) + "\n")
    score = {r["key"]: r.get("cdr_pll") for r in scored}

    # credit per differing residue
    credits: dict[str, dict[int, float]] = defaultdict(dict)
    residue_rows = []
    for pair in pairs:
        pid = f"{pair['case_id']}:{pair['winner_sample_id']}:{pair['loser_sample_id']}"
        r_win = base[pair["winner_sample_id"]]["cdr_pll"]
        r_lose = base[pair["loser_sample_id"]]["cdr_pll"]
        w_rows = {r["position"]: r for r in cf_rows
                  if r["pair_id"] == pid and r["kind"] == "winner_drop"}
        l_rows = {r["position"]: r for r in cf_rows
                  if r["pair_id"] == pid and r["kind"] == "loser_gain"}
        for pos in sorted(w_rows):
            c_drop = r_win - score[w_rows[pos]["cf_id"]]
            c_gain = score[l_rows[pos]["cf_id"]] - r_lose
            c_cons = min(c_drop, c_gain) if (c_drop > 0 and c_gain > 0) else 0.0
            credits[pid][pos] = c_cons
            residue_rows.append({
                "pair_id": pid, "case_id": pair["case_id"], "position": pos,
                "c_drop": c_drop, "c_gain": c_gain, "c_cons": c_cons,
            })
    with (OUT / "counterfactual_credit.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair_id", "case_id", "position",
                                           "c_drop", "c_gain", "c_cons"])
        w.writeheader()
        w.writerows(residue_rows)

    out = {"eta": 0.75, "pairs": {}}
    n_fallback = 0
    for pid, cred in credits.items():
        n = len(cred)
        w, fallback = cf_weights(cred, n)
        n_fallback += int(fallback)
        out["pairs"][pid] = {"case_id": pid.split(":")[0], "n_diff": n,
                             "cf": {str(k): v for k, v in w.items()}}
    (OUT / "weights.json").write_text(json.dumps(out, indent=1))
    cons_all = [r["c_cons"] for r in residue_rows]
    stats = {
        "n_pairs": len(pairs), "n_differing_positions": len(residue_rows),
        "fallback_rate": n_fallback / max(1, len(pairs)),
        "c_cons_positive_fraction": sum(1 for c in cons_all if c > 0) / max(1, len(cons_all)),
        "reward": "esmc_vhh_cdr_pll",
    }
    (OUT / "credit_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
