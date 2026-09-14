#!/usr/bin/env python
"""Phase B query probe: evaluate 60/70/80/90% anchors on train trajectories."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cf_opsd.query_selector import evaluate_candidates, select_query  # noqa: E402
from vhh_rl.cf_opsd.rollout import load_trajectories  # noqa: E402
from vhh_rl.data.case import RLCase  # noqa: E402

CFG = ROOT / "configs/cf_opsd/fixed_cases.yaml"
ROLL = ROOT / "runs/cf_opsd/rollouts"
OUT = ROOT / "runs/cf_opsd"
DOC = ROOT / "docs/cf_opsd/CF_OPSD_TARGET_PROBE.md"


def load_cases() -> dict[str, RLCase]:
    cases = {}
    for line in (ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl").open():
        row = json.loads(line)
        cases[row["case_id"]] = RLCase(
            case_id=row["case_id"], structure_path=Path(row["structure_path"]),
            chain_id=row.get("chain_id", "A"), full_sequence=row["full_sequence"],
            design_positions=tuple(row["design_positions"]),
            fr_positions=tuple(row["fr_positions"]),
            split=row.get("split", "train"), seed_base=int(row.get("seed_base", 0)),
        )
    return cases


def main() -> None:
    cfg = yaml.safe_load(CFG.read_text())
    cases = load_cases()
    per_case = {}
    all_candidates = {}
    for cid in cfg["train_cases"]:
        case = cases[cid]
        seed = case.seed_base + 800000
        trajs = load_trajectories(ROLL / "train" / cid / f"seed{seed}.pt")
        trajs = [t for t in trajs if not t.contains_invalid and t.fr_mismatch == 0]
        cands = evaluate_candidates(trajs, case.design_positions)
        per_case[cid] = [
            {"progress": c.progress, "step": c.step, "valid_rate": c.valid_rate,
             "median_cdr_identity": c.median_cdr_identity,
             "median_full_identity": c.median_full_identity} for c in cands
        ]
        for c in cands:
            all_candidates.setdefault(c.progress, []).append(c)
        print(f"[{cid}] " + " ".join(
            f"q={c.progress:.2f}: valid={c.valid_rate:.2f} cdr={c.median_cdr_identity:.2f}"
            for c in cands))

    # aggregate across cases: median over pooled candidates
    import statistics as st
    agg = []
    for progress in sorted(all_candidates):
        cs = all_candidates[progress]
        agg.append({
            "progress": progress,
            "valid_rate": st.mean(c.valid_rate for c in cs),
            "median_cdr_identity": st.median(c.median_cdr_identity for c in cs),
            "median_full_identity": st.median(c.median_full_identity for c in cs),
        })
    first_ok = next((a for a in agg if a["valid_rate"] >= 0.90 and a["median_cdr_identity"] >= 0.80), None)
    last = agg[-1]
    if first_ok:
        q_star, status = first_ok["progress"], "PASS"
    else:
        q_star = last["progress"]
        status = "QUERY_ANCHOR_WEAK" if last["valid_rate"] >= 0.80 else "NO_GO"

    payload = {"per_case": per_case, "aggregate": agg,
               "q_star_progress": q_star, "status": status}
    (OUT / "query_probe.json").write_text(json.dumps(payload, indent=1))
    lines = "\n".join(
        f"| {a['progress']:.2f} | {a['valid_rate']:.3f} | {a['median_cdr_identity']:.3f} | "
        f"{a['median_full_identity']:.3f} |" for a in agg)
    section = f"""
## Query probe（train 4 cases × K=8）

| progress | valid rate | median CDR identity vs endpoint | median full identity |
|---|---|---|---|
{lines}

- 选择：**q\\* = {q_star:.0%}**（最早满足 valid ≥ 0.90 且 CDR identity ≥ 0.80）
- 状态：**{status}**
"""
    probe_path = OUT / "query_probe_section.md"
    probe_path.write_text(section)
    print(json.dumps({"q_star": q_star, "status": status}, indent=1))
    print(f"wrote {probe_path}")


if __name__ == "__main__":
    main()
