#!/usr/bin/env python
"""CF-DPO v2 experiment 3 driver: legal geometry lift + same-sequence
realizations, with pre-registered acceptance budgets (proposal §12-3)."""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cf_dpo_v2.geometry_lift import (  # noqa: E402
    build_local_lift, build_same_sequence_realizations,
)
from vhh_rl.data.case import RLCase  # noqa: E402

POOL = ROOT / "runs/native_pool"
CF = ROOT / "runs/next_stage/counterfactual"
OUT = ROOT / "runs/cf_dpo_v2/geometry_lift"
DOC = ROOT / "docs/cf_dpo_v2/EXP3_GEOMETRY_LIFT.md"
TOL = 0.05
N_PAIRS = 32
MAX_SITES_PER_PAIR = 3
GATE_ACCEPTANCE = 0.50
GATE_SAME_SEQ_COVERAGE = 0.80


def load_cases() -> dict[str, RLCase]:
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
            seed_base=int(row.get("seed_base", 0)),
        )
    return cases


def load_pool_samples(case_id: str) -> dict[str, dict]:
    out = {}
    for line in (POOL / "train" / case_id / "metadata.jsonl").open():
        r = json.loads(line)
        out[r["sample_id"]] = {
            "sequence": r["decoded_sequence"],
            "coords": torch.load(r["coords_path"], map_location="cpu", weights_only=True).float(),
            "reward": r["reward_raw"],
        }
    return out


def classify(drop: float, gain: float) -> str:
    if drop > TOL and gain > TOL:
        return "both_positive"
    if drop < -TOL and gain < -TOL:
        return "both_negative"
    if drop * gain < 0:
        return "sign_flip"
    return "small"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=N_PAIRS)
    parser.add_argument("--sites-per-pair", type=int, default=MAX_SITES_PER_PAIR)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cases = load_cases()
    credit = defaultdict(dict)
    for row in csv.DictReader((CF / "residue_credit.csv").open()):
        credit[row["pair_id"]][int(row["position"])] = (float(row["c_drop"]),
                                                        float(row["c_gain"]),
                                                        float(row["c_cons"]))
    pairs = [json.loads(l) for l in (POOL / "pairs_train.jsonl").open()]

    # rank pairs by class coverage (prefer flip > both_negative > both_positive)
    def pair_value(pair):
        pid = f"{pair['case_id']}:{pair['winner_sample_id']}:{pair['loser_sample_id']}"
        classes = [classify(*credit[pid][pos][:2]) for pos in credit.get(pid, {})]
        return (classes.count("sign_flip"), classes.count("both_negative"),
                classes.count("both_positive"))
    pairs_sorted = sorted(pairs, key=pair_value, reverse=True)[: args.n_pairs]

    feats_cache: dict[str, dict] = {}
    samples_cache: dict[str, dict[str, dict]] = {}

    def feats_for(case_id):
        if case_id not in feats_cache:
            feats_cache[case_id] = torch.load(
                POOL / "train" / case_id / "feats_common.pt",
                map_location="cpu", weights_only=False)
            samples_cache[case_id] = load_pool_samples(case_id)
        return feats_cache[case_id], samples_cache[case_id]

    attempts = []
    per_class = defaultdict(lambda: {"attempts": 0, "exact": 0})
    per_reason = defaultdict(int)
    lifted_sequences: dict[str, torch.Tensor] = {}
    for pair in pairs_sorted:
        case = cases[pair["case_id"]]
        pid = f"{pair['case_id']}:{pair['winner_sample_id']}:{pair['loser_sample_id']}"
        if pid not in credit:
            continue
        feats, samples = feats_for(case.case_id)
        winner = samples[pair["winner_sample_id"]]
        loser = samples[pair["loser_sample_id"]]
        by_class = defaultdict(list)
        for pos, (d, g, _cc) in credit[pid].items():
            by_class[classify(d, g)].append(pos)
        sites = []
        for cls in ("sign_flip", "both_negative", "both_positive"):
            if by_class[cls]:
                sites.append(by_class[cls][0])
        for cls in ("sign_flip", "both_negative", "both_positive"):
            for pos in by_class[cls][1:]:
                if len(sites) >= args.sites_per_pair:
                    break
                sites.append(pos)
        sites = sites[: args.sites_per_pair]
        for pos in sites:
            cls = classify(*credit[pid][pos][:2])
            w_seq, l_seq = winner["sequence"], loser["sequence"]
            if w_seq[pos] == l_seq[pos]:
                continue
            for direction, acceptor, donor, expected, other, anchor in (
                ("winner_drop", winner["coords"], loser["coords"],
                 w_seq[:pos] + l_seq[pos] + w_seq[pos + 1:], l_seq, w_seq),
                ("loser_gain", loser["coords"], winner["coords"],
                 l_seq[:pos] + w_seq[pos] + l_seq[pos + 1:], w_seq, l_seq),
            ):
                att = build_local_lift(acceptor, donor, feats, pos,
                                       case.full_sequence, case.fr_positions,
                                       expected, other, anchor)
                per_class[cls]["attempts"] += 1
                per_class[cls]["exact"] += int(att.ok)
                per_reason[att.reason] += 1
                row = {"case_id": case.case_id, "pair_id": pid, "position": pos,
                       "class": cls, "direction": direction, "ok": att.ok,
                       "reason": att.reason,
                       "acceptor_aa": (w_seq if direction == "winner_drop" else l_seq)[pos],
                       "donor_aa": (l_seq if direction == "winner_drop" else w_seq)[pos],
                       "moved_rms": att.moved_rms, "touched": att.touched_atoms}
                attempts.append(row)
                if att.ok and att.coords is not None:
                    key = f"{case.case_id}:{expected}"
                    lifted_sequences.setdefault(key, att.coords)
    n_attempts = len(attempts)
    n_exact = sum(1 for a in attempts if a["ok"])
    acceptance = n_exact / max(1, n_attempts)
    fr_bad = sum(1 for a in attempts if a["reason"] == "fr_changed")

    # same-sequence M=2 realizations
    same_seq = []
    for key, coords in list(lifted_sequences.items())[:64]:
        case_id, seq = key.split(":", 1)
        feats, _samples = feats_for(case_id)
        case = cases[case_id]
        reals = build_same_sequence_realizations(
            coords, feats, seq, case.fr_positions, case.full_sequence,
            n_realizations=2, sigma=0.08, max_tries=40, seed=hash(key) % 10_000)
        same_seq.append({"key": key, "n_realizations": len(reals)})
    coverage = sum(1 for r in same_seq if r["n_realizations"] >= 2) / max(1, len(same_seq))

    gate = {
        "acceptance_rate": acceptance,
        "fr_mismatch_attempts": fr_bad,
        "same_seq_coverage": coverage,
        "pass": (acceptance >= GATE_ACCEPTANCE and fr_bad == 0
                 and coverage >= GATE_SAME_SEQ_COVERAGE),
    }
    summary = {"n_pairs": len(pairs_sorted), "n_attempts": n_attempts,
               "n_exact": n_exact, "acceptance": acceptance,
               "per_class": {k: dict(v) for k, v in per_class.items()},
               "reasons": dict(per_reason), "same_seq_coverage": coverage,
               "gate": gate}
    (OUT / "lift_summary.json").write_text(json.dumps(summary, indent=1))
    with (OUT / "lift_attempts.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(attempts[0].keys()))
        w.writeheader()
        w.writerows(attempts)
    (OUT / "same_seq.json").write_text(json.dumps(same_seq, indent=1))

    rows = "\n".join(
        f"| {k} | {v['attempts']} | {v['exact']} | {v['exact'] / max(1, v['attempts']):.2f} |"
        for k, v in sorted(per_class.items()))
    reason_rows = "\n".join(f"| {k} | {v} |" for k, v in sorted(per_reason.items()))
    doc = f"""# CF-DPO v2 — Experiment 3: legal native geometry lift

预登记预算：尝试 {len(pairs_sorted)} 对训练样本（每对最多 {args.sites_per_pair} 个位点、双向），
接受阈值 acceptance ≥ {GATE_ACCEPTANCE:.0%}、FR mismatch = 0、同序列 M=2 覆盖率 ≥ {GATE_SAME_SEQ_COVERAGE:.0%}。

## 结果

- 尝试边：{n_attempts}；精确 lift：{n_exact}（acceptance = {acceptance:.1%}）
- FR mismatch 尝试：{fr_bad}
- 同序列 M=2 覆盖率：{coverage:.1%}

### 按符号类别

| class | attempts | exact | rate |
|---|---|---|---|
{rows}

### 失败原因

| reason | count |
|---|---|
{reason_rows}

## Gate（§14：合法几何补全不稳定则不进入训练）

**{'PASS' if gate['pass'] else 'FAIL'}** — acceptance={acceptance:.1%}，
FR mismatch={fr_bad}，same-seq coverage={coverage:.1%}。

失败样本未静默丢弃：逐边记录在 `lift_attempts.csv`（含类别、残基对、位移 RMS、失败原因）。
"""
    DOC.write_text(doc)
    print(json.dumps(summary, indent=1))
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
