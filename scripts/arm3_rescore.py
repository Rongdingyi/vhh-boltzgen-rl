#!/usr/bin/env python3
"""Full-panel rescoring for the arm comparison (base IF / RL / G0 / M1-beta1).

Loads the frozen VHH-source ensemble once and dumps the complete
VHHSourceResult dict for every sequence. GPU job (ESM-C embeddings).

Modes:
  (default)  collect base+rl (v100eval CSVs) + g0 top-8, write arm3_scores.jsonl
  extra      score m1b1_jobs.json (m1 + m1ab beta=1.0, 8/case), append
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
RL_SRC = ROOT / "src"
SCORER_ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_guidance")
G0_DIR = Path("/share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance/outputs/cdr_all/g0_t1.00/g0_scores")
OUT = ROOT / "runs/round1_rl_split/arm3_scores.jsonl"
EXTRA_JOBS = ROOT / "runs/round1_rl_split/m1b1_jobs.json"

sys.path.insert(0, str(RL_SRC))
sys.path.insert(0, str(SCORER_ROOT / "src"))
from vhh_guidance.adapters.vhh_source_adapter import VHHSourceAdapter
from vhh_guidance.schemas import VHHSpec
from vhh_rl.rewards.scorer_worker import build_spec
from vhh_rl.cli.common import cdr_range
from vhh_rl.data.case import RLCase


def spec_for(manifest_row: dict) -> dict:
    case = RLCase(
        case_id=manifest_row["case_id"],
        structure_path=Path(manifest_row["structure_path"]),
        chain_id=manifest_row.get("chain_id", "A"),
        full_sequence=manifest_row["full_sequence"],
        design_positions=tuple(manifest_row["design_positions"]),
        fr_positions=tuple(manifest_row.get("fr_positions", ())),
        split=manifest_row.get("split", "test"),
        seed_base=int(manifest_row.get("seed_base", 0)),
    )
    return {
        "sequence": case.full_sequence,
        "design_positions": list(case.design_positions),
        "cdr1": list(cdr_range(case, "cdr1")),
        "cdr2": list(cdr_range(case, "cdr2")),
        "cdr3": list(cdr_range(case, "cdr3")),
        "chain_id": case.chain_id,
    }


def collect_base_jobs() -> list[tuple[str, str, int, str]]:
    jobs: list[tuple[str, str, int, str]] = []
    for i in range(3):
        with open(ROOT / f"runs/round1_rl_split/v100eval_shard{i}/eval_per_case.csv") as f:
            for r in csv.DictReader(f):
                jobs.append((r["case_id"], r["arm"], -1, r["sequence"]))
    counters: dict[tuple[str, str], int] = {}
    out = []
    for c, a, _, s in jobs:
        k = (c, a)
        out.append((c, a, counters.get(k, 0), s))
        counters[k] = counters.get(k, 0) + 1
    # G0 top-8 per case
    g0rows = []
    for fp in sorted(glob.glob(str(G0_DIR / "g0_scores_shard*.csv"))):
        with open(fp) as f:
            for r in csv.DictReader(f):
                if str(r.get("selected_in_top8", "")).strip().lower() == "true":
                    g0rows.append(r)
    by_case: dict[str, list] = {}
    for r in g0rows:
        by_case.setdefault(r["case_id"], []).append(r)
    for c, rs in by_case.items():
        rs.sort(key=lambda r: int(r["selection_rank"] or 999))
        for i, r in enumerate(rs[:8]):
            out.append((c, "g0", i, r["sequence"]))
    return out


def main(extra_only: bool = False) -> None:
    manifest = {}
    for line in open(ROOT / "runs/round1_rl_split/rl_manifest_valid100test.jsonl"):
        r = json.loads(line)
        manifest[r["case_id"]] = r

    if extra_only:
        jobs = [tuple(j) for j in json.load(open(EXTRA_JOBS))]
        print(f"extra jobs: {len(jobs)}", flush=True)
    else:
        jobs = collect_base_jobs()
        print(f"base jobs: {len(jobs)}", flush=True)

    config = yaml.safe_load((SCORER_ROOT / "configs" / "models.local.yaml").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = VHHSourceAdapter(config, device)
    scorer.load()

    current_key = None
    current_spec = None
    out = open(OUT, "a" if extra_only else "w")
    done = 0
    with torch.no_grad():
        jobs.sort(key=lambda j: j[0])
        buf: list[tuple[str, str, int, str]] = []
        last_case = None

        def flush():
            nonlocal buf, current_key, current_spec, done
            if not buf:
                return
            c = buf[0][0]
            row = spec_for(manifest[c])
            key = json.dumps(row, sort_keys=True)
            if key != current_key:
                current_spec = build_spec(row, VHHSpec)
                current_key = key
            results = scorer.score_sequences([s for _, _, _, s in buf], current_spec)
            for (cc, arm, idx, seq), res in zip(buf, results):
                out.write(json.dumps({"case_id": cc, "arm": arm, "idx": idx,
                                      "sequence": seq, **res.as_dict()}) + "\n")
            done += len(buf)
            buf = []
            if done % 400 == 0:
                print(f"  scored {done}/{len(jobs)}", flush=True)

        for job in jobs:
            if job[0] != last_case:
                flush()
                last_case = job[0]
            buf.append(job)
        flush()
    out.close()
    print(f"DONE {done} rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main(extra_only=(len(sys.argv) > 1 and sys.argv[1] == "extra"))
