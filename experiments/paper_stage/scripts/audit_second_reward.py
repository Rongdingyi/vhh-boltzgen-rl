#!/usr/bin/env python
"""Paper-stage Step 6: second black-box reward audit (task book §37-§40).

Candidate A: VHH-ESM-C exact masked CDR PLL (sequence-only, architecture
different from the CDR classifier; no new labels).

Audits on the 24-case train base pool (768 sequences):
  variance / median per-case std / corr with cdr_camelid_margin / ranking.
Gate: median per-case std > 1e-4 AND |Spearman(classifier, second)| < 0.90.

Writes runs/paper_stage/second_reward/audit_base_pool.csv + audit_stats.json
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
GUID = Path("/share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance")
ESM = Path("/share/home/rongdingyi/programs/proteingen/esm")
for p in (str(GUID / "src"), str(ESM), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = ROOT / "runs/paper_stage/second_reward"
MODEL = "/share/data/limc/esmc-600m-vhh"


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return float(r.statistic)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {}
    for line in (ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl").open():
        row = json.loads(line)
        if row.get("split") != "train":
            continue
        cases[row["case_id"]] = {
            "native_sequence": row["full_sequence"],
            "design_positions": [int(p) for p in row["design_positions"]],
        }
    candidates, meta = [], []
    for meta_path in sorted((ROOT / "runs/native_pool/train").glob("*/metadata.jsonl")):
        for line in meta_path.open():
            r = json.loads(line)
            if r["reward_raw"] is None or r["contains_UNK"]:
                continue
            candidates.append({"case_id": r["case_id"], "sequence": r["decoded_sequence"]})
            meta.append(r)
    print(f"{len(candidates)} sequences from {len({m['case_id'] for m in meta})} cases")

    from esmc_reward import ESMCReward, load_cases

    scorer = ESMCReward(cache_path=OUT / "esmc_cache.sqlite")
    records = [{"key": f"{c['case_id']}|{i}", "case_id": c["case_id"], "sequence": c["sequence"]}
               for i, c in enumerate(candidates)]
    cases_all = load_cases("train")
    scored = scorer.score(records, cases_all, batch_size=32)

    rows = []
    for r, c in zip(scored, candidates):
        if r.get("status") != "PASS" or r.get("cdr_pll") is None:
            continue
        rows.append({
            "case_id": c["case_id"], "sequence": c["sequence"],
            "cdr_pll": float(r["cdr_pll"]), "full_pll": float(r["full_pll"]),
        })
    # join classifier reward by sequence (per case)
    reward = {}
    for m in meta:
        reward[(m["case_id"], m["decoded_sequence"])] = float(m["reward_raw"])
    xs, ys, per_case = [], [], {}
    for row in rows:
        cl = reward.get((row["case_id"], row["sequence"]))
        if cl is None:
            continue
        xs.append(cl)
        ys.append(row["cdr_pll"])
        per_case.setdefault(row["case_id"], {"cl": [], "pll": []})
        per_case[row["case_id"]]["cl"].append(cl)
        per_case[row["case_id"]]["pll"].append(row["cdr_pll"])
    global_spearman = spearman(xs, ys) if len(xs) > 5 else None
    per_case_std = [st.stdev(v["pll"]) for v in per_case.values() if len(v["pll"]) > 1]
    per_case_spearman = [spearman(v["cl"], v["pll"]) for v in per_case.values()
                         if len(v["cl"]) > 3 and len(set(v["cl"])) > 1
                         and len(set(v["pll"])) > 1]
    stats = {
        "n_sequences": len(xs),
        "n_cases": len(per_case),
        "cdr_pll_mean": float(np.mean(ys)),
        "cdr_pll_std": float(np.std(ys)),
        "median_per_case_std": st.median(per_case_std) if per_case_std else None,
        "global_spearman_classifier_vs_cdr_pll": global_spearman,
        "median_per_case_spearman": st.median(per_case_spearman) if per_case_spearman else None,
        "fraction_cases_abs_spearman_lt_0.9": (
            sum(1 for r in per_case_spearman if abs(r) < 0.9) / len(per_case_spearman)
            if per_case_spearman else None),
        "gate_median_std_gt_1e-4": bool(per_case_std and st.median(per_case_std) > 1e-4),
        "gate_spearman_lt_0.90": bool(global_spearman is not None and abs(global_spearman) < 0.90),
    }
    stats["second_reward_selected"] = stats["gate_median_std_gt_1e-4"] and stats["gate_spearman_lt_0.90"]
    with (OUT / "audit_base_pool.csv").open("w") as fh:
        fh.write("case_id,sequence,cdr_pll,full_pll,classifier_reward\n")
        for row in rows:
            cl = reward.get((row["case_id"], row["sequence"]))
            fh.write(f"{row['case_id']},{row['sequence']},{row['cdr_pll']:.6f},"
                     f"{row['full_pll']:.6f},{cl if cl is not None else ''}\n")
    (OUT / "audit_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
