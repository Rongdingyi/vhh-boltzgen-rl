#!/usr/bin/env python
"""Gate 2: build distillation records + carrier audit (§28-§30/§34)."""
from __future__ import annotations

import argparse
import csv
import json
import sys

import torch

torch.set_float32_matmul_precision("high")  # same-query replay requires the sampler precision

import _common as C  # noqa: E402
from vhh_rl.branch_distill.behavior_loop import Gate3Config, build_online_records
from vhh_rl.branch_distill.types import BranchGroup, SiblingEndpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--override-gate", default=None)
    parser.add_argument("--records", type=int, default=12)
    args = parser.parse_args()
    C.require_gate(C.GATE1_DIR / "gate1.json", "pass", True, args.override_gate, "Gate 1")
    C.GATE2_DIR.mkdir(parents=True, exist_ok=True)
    selected = json.loads((C.GATE1_DIR / "selected_progress.json").read_text())
    cases = C.load_cases(C.TRAIN_CASES)
    payload = torch.load(C.GATE1_DIR / "branch_groups.pt", map_location="cpu",
                         weights_only=False)
    groups = [_group_from_row(row) for row in payload["groups"]]
    groups = [g for g in groups if abs(g.progress - selected["selected_progress"]) < 1e-9]

    all_records, all_audits = [], []
    cfg = Gate3Config(train_cases=C.TRAIN_CASES)
    for case_id in C.TRAIN_CASES:
        case = cases[case_id]
        sub = [g for g in groups if g.case_id == case_id]
        records, audits = build_online_records(
            sub, cfg, case.design_positions, case.full_sequence, case.fr_positions,
            peer_types=("best_vs_worst", "best_vs_median"))
        all_records += records
        all_audits += audits

    # §34 fixed selection: sort by (case_id, source_idx, peer_type), >=2 per case
    # deterministic order, no reward-gap cherry-picking
    all_records.sort(key=lambda r: (r.case_id, r.meta["peer_type"]))
    selected_records, counts = [], {}
    for record in all_records:
        if counts.get(record.case_id, 0) >= 2:
            continue
        selected_records.append(record)
        counts[record.case_id] = counts.get(record.case_id, 0) + 1
        if len(selected_records) >= args.records:
            break
    if len(selected_records) < args.records:
        chosen = {id(r) for r in selected_records}
        for record in all_records:
            if len(selected_records) >= args.records:
                break
            if id(record) not in chosen:
                selected_records.append(record)
                chosen.add(id(record))
    torch.save({"records": [r.__dict__ for r in selected_records]},
               C.GATE2_DIR / "records.pt")

    fields = sorted({k for row in all_audits for k in row})
    with (C.GATE2_DIR / "target_audit.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_audits)
    eligible_rows = [a for a in all_audits if a.get("eligible")]
    # Amendment 1: the gate is that every eligible record keeps at least one
    # carrier-verified position; the full-set carrier decode is diagnostic.
    # §30/§39 B: the endpoint-based carrier is the pre-registered legality probe
    # (the early query anchor may itself be invalid).  Amendment 1 limits
    # supervision to the carrier-matched positions and reports the original
    # per-position fidelity; the target's own decode is kept as a diagnostic.
    carrier_ok = all(a.get("carrier_verified_positions", 0) > 0
                     and a.get("carrier_fr", 0) == 0
                     for a in eligible_rows)
    summary = {
        "protocol_sha256": C.protocol_hash(),
        "selected_progress": selected["selected_progress"],
        "n_records": len(selected_records),
        "records_per_case": counts,
        "n_audit_rows": len(all_audits),
        "n_eligible_rows": len(eligible_rows),
        "carrier_pass": bool(carrier_ok),
        "carrier_original_pooled_rate": _pooled_rate(all_audits),
        "n_rows_original_carrier_invalid": sum(
            1 for a in eligible_rows if a.get("carrier_invalid")),
        "n_rows_target_failed_diagnostic": sum(
            1 for a in eligible_rows if not a.get("target_all_match")),
        "n_rows_target_invalid_diagnostic": sum(
            1 for a in eligible_rows if a.get("target_invalid")),
        "amendment_id": 1,
        "amendment_sha256": C.sha256(C.CONFIG_DIR / "AMENDMENT_1_VERIFIED_POSITIONS.yaml"),
    }
    C.write_json(C.GATE2_DIR / "record_build.json", summary)
    print(json.dumps(summary, indent=1))
    if not carrier_ok:
        sys.exit(1)   # §30: STOP, no radius sweep


def _pooled_rate(audits: list[dict]) -> float | None:
    import ast
    total = matched = 0
    for audit in audits:
        matches = audit.get("carrier_matches")
        if isinstance(matches, str):
            matches = ast.literal_eval(matches)
        if not matches:
            continue
        total += len(matches)
        matched += sum(1 for v in matches.values() if v)
    return (matched / total) if total else None


def _group_from_row(row: dict) -> BranchGroup:
    siblings = [SiblingEndpoint(**s) if isinstance(s, dict) else s
                for s in row["siblings"]]
    row = dict(row)
    row["siblings"] = siblings
    known = {f for f in BranchGroup.__dataclass_fields__}
    return BranchGroup(**{k: v for k, v in row.items() if k in known})


if __name__ == "__main__":
    main()
