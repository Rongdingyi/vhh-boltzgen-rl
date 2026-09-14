#!/usr/bin/env python
"""Phase B: winner/loser credit + bounded target construction + Gate B.

Builds CF targets at rho = 0.25/0.50/1.00 and (at the selected radius) the
no-op / random / shuffled controls, decodes every target with the official
res_from_atom14, scores it with the classifier, and writes the Gate B report.
"""
from __future__ import annotations

import csv
import json
import random
import statistics as st
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cf_opsd.credit import pair_credit  # noqa: E402
from vhh_rl.cf_opsd.rollout import load_trajectories  # noqa: E402
from vhh_rl.cf_opsd.target_builder import build_target  # noqa: E402
from vhh_rl.cf_opsd.target_metrics import (  # noqa: E402
    credit_match_rate, decode_and_score, third_aa_rate,
)
from vhh_rl.cli.common import case_spec_for  # noqa: E402
from vhh_rl.data.case import RLCase  # noqa: E402
from vhh_rl.native_atom14.decode import decode_atom14, fr_check, sequence_from_feat  # noqa: E402
from vhh_rl.native_atom14.reward import make_reward_adapter  # noqa: E402

CFG = ROOT / "configs/cf_opsd/fixed_cases.yaml"
ROLL = ROOT / "runs/cf_opsd/rollouts"
OUT = ROOT / "runs/cf_opsd/target_probe"
DOC = ROOT / "docs/cf_opsd/CF_OPSD_TARGET_PROBE.md"
RADII = (0.25, 0.50, 1.00)
CONTROLS = ("noop", "random", "shuffle")


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


def decode_seq(coords, feats) -> tuple[str, bool]:
    feat = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in feats.items()}
    feat["coords"] = coords.clone()
    out = decode_atom14(feat)
    seq, _tok, invalid = sequence_from_feat(out)
    return seq, invalid


def main() -> None:
    cfg = yaml.safe_load(CFG.read_text())
    query = json.loads((ROOT / "runs/cf_opsd/query_probe.json").read_text())
    q_star = float(query["q_star_progress"])
    status = query["status"]
    if status == "NO_GO":
        (DOC.parent / "CF_OPSD_TARGET_PROBE.md").write_text(
            "# CF-OPSD Target Probe\n\nQuery probe NO_GO（90% anchor valid < 0.8）→ 停止。\n")
        print("QUERY NO_GO -> STOP")
        return
    cases = load_cases()
    scorer = make_reward_adapter(cache_path=ROOT / "runs/cf_opsd/reward_cache.sqlite")

    rows = []
    targets_store = []
    for cid in cfg["train_cases"]:
        case = cases[cid]
        seed = case.seed_base + 800000
        trajs = [t for t in load_trajectories(ROLL / "train" / cid / f"seed{seed}.pt")
                 if t.endpoint_reward is not None and not t.contains_invalid and t.fr_mismatch == 0]
        ranked = sorted(trajs, key=lambda t: t.endpoint_reward, reverse=True)
        winner, loser = ranked[0], ranked[-1]
        n_steps = len(loser.sigmas)
        step = max(0, min(n_steps - 1, int(round(q_star * n_steps)) - 1))
        anchor = loser.anchors[step]
        anchor_seq, anchor_invalid = decode_seq(anchor, loser.feats_common)
        anchor_reward = None
        if not anchor_invalid:
            batch = scorer.score_sequences([anchor_seq], spec=case_spec_for(case))
            anchor_reward = float(batch.raw_scores[0])

        credit = pair_credit(scorer, case, f"{cid}_w", f"{cid}_l",
                             winner.endpoint_sequence, loser.endpoint_sequence,
                             case.design_positions, case.fr_positions)
        credits = credit["credits"]
        differing = sorted(credits)
        print(f"[{cid}] winner={winner.endpoint_reward:+.2f} loser={loser.endpoint_reward:+.2f} "
              f"anchor_reward={anchor_reward} n_diff={len(differing)} "
              f"n_credit>0={sum(1 for c in credits.values() if c>0)}")

        for radius in RADII:
            bt = build_target(anchor, winner.endpoint_coords, loser.feats_common,
                              differing, credits, radius, control="cf")
            metrics = decode_and_score(bt.target_coords, loser.feats_common,
                                       case.full_sequence, case.fr_positions, scorer,
                                       case, winner.endpoint_reward, anchor_reward,
                                       case.design_positions)
            credited = [p for p, c in credits.items() if c > 0]
            match = credit_match_rate(metrics["target_sequence"], winner.endpoint_sequence, credited)
            third = third_aa_rate(metrics["target_sequence"], winner.endpoint_sequence,
                                  loser.endpoint_sequence, anchor_seq, differing)
            row = {
                "case_id": cid, "control": "cf", "radius": radius,
                "anchor_reward": anchor_reward, "target_reward": metrics["target_reward"],
                "winner_reward": winner.endpoint_reward, "loser_reward": loser.endpoint_reward,
                "target_minus_anchor": metrics["target_minus_anchor"],
                "winner_minus_anchor": winner.endpoint_reward - (anchor_reward or 0.0),
                "credit_match": match, "third_aa_rate": third,
                "invalid": metrics["target_invalid"], "fr_mismatch": metrics["target_fr_mismatch"],
                "actual_rms": bt.actual_rms, "saturation": bt.saturation,
                "n_diff": len(differing),
                "n_credit": len(credited),
                "credit_mass": sum(credits.values()),
            }
            rows.append(row)
            targets_store.append({
                "case_id": cid, "control": "cf", "radius": radius,
                "query_index": step, "query_progress": q_star,
                "sigma": loser.sigmas[step],
                "query_coords": loser.query_states[step],
                "anchor_coords": anchor,
                "winner_coords": winner.endpoint_coords,
                "loser_coords": loser.endpoint_coords,
                "winner_sequence": winner.endpoint_sequence,
                "loser_sequence": loser.endpoint_sequence,
                "anchor_sequence": anchor_seq,
                "anchor_reward": anchor_reward,
                "credits": credits,
                "target_coords": bt.target_coords,
                "target_sequence": metrics["target_sequence"],
                "target_reward": metrics["target_reward"],
                "feats_common": loser.feats_common,
            })

    # radius summary + Gate B
    radius_rows = []
    for radius in RADII:
        rr = [r for r in rows if r["radius"] == radius and r["control"] == "cf"]
        gains = [r["target_minus_anchor"] for r in rr if r["target_minus_anchor"] is not None]
        matches = [r["credit_match"] for r in rr if r["credit_match"] is not None]
        radius_rows.append({
            "radius": radius,
            "n": len(rr),
            "invalid_rate": sum(1 for r in rr if r["invalid"]) / len(rr),
            "fr_mismatch_total": sum(r["fr_mismatch"] for r in rr),
            "median_target_minus_anchor": st.median(gains) if gains else None,
            "positive_cases": sum(1 for g in gains if g > 0),
            "median_credit_match": st.median(matches) if matches else None,
            "median_third_aa_rate": st.median([r["third_aa_rate"] for r in rr
                                               if r["third_aa_rate"] is not None]),
            "median_actual_rms": st.median([r["actual_rms"] for r in rr]),
        })
    gate = {}
    for r in radius_rows:
        gate[r["radius"]] = (
            r["invalid_rate"] <= 0.05
            and r["fr_mismatch_total"] == 0
            and (r["median_target_minus_anchor"] or 0) > 0
            and (r["median_credit_match"] or 0) >= 0.50
            and r["positive_cases"] >= 3
        )
    passing = [r for r in radius_rows if gate[r["radius"]]]
    selected_radius = min((r["radius"] for r in passing), default=None)

    # controls at selected radius
    control_rows = []
    if selected_radius is not None:
        rng = random.Random(1234)
        for cid in cfg["train_cases"]:
            case = cases[cid]
            seed = case.seed_base + 800000
            trajs = [t for t in load_trajectories(ROLL / "train" / cid / f"seed{seed}.pt")
                     if t.endpoint_reward is not None and not t.contains_invalid and t.fr_mismatch == 0]
            ranked = sorted(trajs, key=lambda t: t.endpoint_reward, reverse=True)
            winner, loser = ranked[0], ranked[-1]
            n_steps = len(loser.sigmas)
            step = max(0, min(n_steps - 1, int(round(q_star * n_steps)) - 1))
            anchor = loser.anchors[step]
            anchor_seq, _inv = decode_seq(anchor, loser.feats_common)
            anchor_reward = None
            batch = scorer.score_sequences([anchor_seq], spec=case_spec_for(case))
            anchor_reward = float(batch.raw_scores[0])
            credit = pair_credit(scorer, case, f"{cid}_w", f"{cid}_l",
                                 winner.endpoint_sequence, loser.endpoint_sequence,
                                 case.design_positions, case.fr_positions)
            credits = credit["credits"]
            differing = sorted(credits)
            cf_row = next(r for r in rows if r["case_id"] == cid
                          and r["radius"] == selected_radius and r["control"] == "cf")
            control_rows.append({**cf_row, "control": "cf"})
            for control in CONTROLS:
                bt = build_target(anchor, winner.endpoint_coords, loser.feats_common,
                                  differing, credits, selected_radius,
                                  control=control, rng=rng)
                seq, invalid = decode_seq(bt.target_coords, loser.feats_common)
                reward = None
                if not invalid:
                    fr = fr_check(seq, case.full_sequence, case.fr_positions)
                    if fr["fr_mismatch_count"] == 0:
                        batch = scorer.score_sequences([seq], spec=case_spec_for(case))
                        reward = float(batch.raw_scores[0])
                control_rows.append({
                    "case_id": cid, "control": control, "radius": selected_radius,
                    "target_reward": reward,
                    "target_minus_anchor": (reward - anchor_reward) if reward is not None else None,
                    "credit_match": credit_match_rate(seq, winner.endpoint_sequence,
                                                      [p for p, c in credits.items() if c > 0]),
                    "invalid": invalid,
                })
    scorer.close()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "targets.pt").write_bytes(b"")  # placeholder replaced below
    torch.save(targets_store, OUT / "targets.pt")
    with (OUT / "target_sequences.jsonl").open("w") as fh:
        for t in targets_store:
            fh.write(json.dumps({k: v for k, v in t.items()
                                 if k not in ("query_coords", "anchor_coords",
                                              "winner_coords", "loser_coords",
                                              "target_coords", "feats_common")}) + "\n")
    with (OUT / "target_metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (OUT / "radius_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(radius_rows[0].keys()))
        w.writeheader()
        w.writerows(radius_rows)
    with (OUT / "control_metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(control_rows[0].keys()))
        w.writeheader()
        w.writerows(control_rows)

    summary = {"q_star_progress": q_star, "query_status": status,
               "radii": radius_rows, "gate_b": gate, "selected_radius": selected_radius}
    (OUT / "gate_b.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))

    # doc
    radius_table = "\n".join(
        f"| {r['radius']} | {r['median_target_minus_anchor']:+.3f} | {r['positive_cases']}/4 | "
        f"{r['median_credit_match']:.2f} | {r['invalid_rate']:.2f} | {r['fr_mismatch_total']} | "
        f"{'PASS' if gate[r['radius']] else 'FAIL'} |" for r in radius_rows)
    control_table = "\n".join(
        f"| {r['control']} | {r['target_minus_anchor']:+.3f} | {r['credit_match']:.2f} |"
        for r in control_rows) if control_rows else ""
    section = f"""## Target construction（q* = {q_star:.0%}）

### Radius sweep（CF target）

| rho (A) | median target-anchor | positive cases | median credit match | invalid | FR mismatch | Gate B |
|---|---|---|---|---|---|---|
{radius_table}

选择半径（满足 Gate 的最小值）：**{selected_radius}**

### Controls（selected radius）

| control | target-anchor | credit match |
|---|---|---|
{control_table}

- Gate B：**{'PASS' if selected_radius is not None else 'FAIL'}**
"""
    probe_section = (ROOT / "runs/cf_opsd/query_probe_section.md").read_text() \
        if (ROOT / "runs/cf_opsd/query_probe_section.md").is_file() else ""
    header = "# CF-OPSD Target Probe\n\n"
    DOC.write_text(header + probe_section + "\n" + section)
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
