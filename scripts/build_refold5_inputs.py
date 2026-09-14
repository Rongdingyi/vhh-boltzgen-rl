#!/usr/bin/env python3
"""Build refold inputs for the 5-arm self-consistency eval.

20 valid100 cases (10 hard + 10 easy by base nativeness) x 5 arms
(base, r1, r2s1, g0, m1-beta1) x 4 sequences = 400 folding inputs.
Writes design_dir/{id}.cif+.npz + manifest.csv, reusing
vhh_esmc_guidance/scripts/cdr_all/build_refold_inputs.write_job.
"""
from __future__ import annotations

import csv
import collections
import glob
import json
import statistics as st
import sys
from pathlib import Path

GUID = Path("/share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance/scripts/cdr_all")
sys.path.insert(0, str(GUID))
from build_refold_inputs import Job, write_job
from common_cdr_all import load_cases

RL = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl/runs/round1_rl_split")
G0_DIR = Path("/share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance/outputs/cdr_all/g0_t1.00/g0_scores")
M1_CSV = "/share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance/outputs/cdr_all/m1_sweep/beta_1.00/m1_beta1.00_shard*.csv"
OUT = RL / "refold5_design"
N_PER_ARM = 4


def main() -> None:
    # sequences from the arm-3 full-panel rescoring
    armseq: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    tmp: dict[tuple[str, str], list[tuple[int, str]]] = collections.defaultdict(list)
    for line in open(RL / "arm3_scores.jsonl"):
        r = json.loads(line)
        if r["arm"] in ("base", "rl", "r2s1"):
            tmp[(r["case_id"], r["arm"])].append((r["idx"], r["sequence"]))
    base_mean = {}
    for (c, a), v in tmp.items():
        v.sort()
        armseq[(c, a)] = [s for _, s in v[:N_PER_ARM]]
        if a == "base":
            # nativeness means recomputed from the same rows is overkill;
            # use file order proxy: recompute from eval CSVs instead (below)
            pass
    # base nativeness per case from the v100 eval CSVs
    bnat: dict[str, list[float]] = collections.defaultdict(list)
    for i in range(3):
        with open(RL / f"v100eval_shard{i}/eval_per_case.csv") as f:
            for r in csv.DictReader(f):
                if r["arm"] == "base":
                    bnat[r["case_id"]].append(float(r["raw_score"]))
    bmean = {c: st.mean(v) for c, v in bnat.items()}
    ranked = sorted(bmean, key=bmean.get)
    hard = ranked[:10]
    easy = ranked[-10:]
    cases = hard + easy
    print("hard:", [f"{c}={bmean[c]:.3f}" for c in hard], flush=True)
    print("easy:", [f"{c}={bmean[c]:.3f}" for c in easy], flush=True)

    # g0 top-4 by selection rank
    g0: dict[str, list[str]] = collections.defaultdict(list)
    rows = []
    for fp in sorted(glob.glob(str(G0_DIR / "g0_scores_shard*.csv"))):
        with open(fp) as f:
            for r in csv.DictReader(f):
                if str(r.get("selected_in_top8", "")).strip().lower() == "true":
                    rows.append(r)
    byc: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        byc[r["case_id"]].append(r)
    for c, rs in byc.items():
        rs.sort(key=lambda r: int(r["selection_rank"] or 999))
        g0[c] = [r["sequence"] for r in rs[:N_PER_ARM]]

    # m1 beta1 first 4 samples
    m1: dict[str, list[str]] = collections.defaultdict(list)
    mrows: dict[str, list] = collections.defaultdict(list)
    for fp in sorted(glob.glob(M1_CSV)):
        with open(fp) as f:
            for r in csv.DictReader(f):
                mrows[r["case_id"]].append(r)
    for c, rs in mrows.items():
        rs.sort(key=lambda r: int(r["sample_index"]))
        m1[c] = [r["sequence"] for r in rs[:N_PER_ARM]]

    loaded = {c.case_id: c for c in load_cases(cases)}
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    tag = {"base": "b", "rl": "r1", "r2s1": "r2", "g0": "g0", "m1": "m1"}
    seqs = {"base": armseq, "rl": armseq, "r2s1": armseq}
    n = 0
    for ci, c in enumerate(cases):
        case = loaded[c]
        for arm in ["base", "rl", "r2s1", "g0", "m1"]:
            if arm in seqs:
                pool = armseq.get((c, arm), [])
            else:
                pool = (g0 if arm == "g0" else m1).get(c, [])
            assert len(pool) >= N_PER_ARM, (c, arm, len(pool))
            for k in range(N_PER_ARM):
                sid = f"c{ci:02d}{tag[arm]}{k}"
                job = Job(sample_id=sid, case_id=c, label=arm, beta="", rank=str(k),
                          sequence=pool[k])
                write_job(job, case, OUT, strip_design_sidechains=True)
                manifest.append({"sample_id": sid, "case_id": c, "arm": arm,
                                 "idx": k})
                n += 1
    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "case_id", "arm", "idx"])
        w.writeheader()
        w.writerows(manifest)
    print(f"DONE {n} refold inputs -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
