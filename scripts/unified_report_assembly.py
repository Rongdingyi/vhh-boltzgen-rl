#!/usr/bin/env python
"""Assemble the unified method comparison table (legacy + RL methods).

Merges:
  - sequence metrics: runs/report_unified/unified_summary.json (PLL/MPNN/recovery)
  - classifier panel: arm3 (G0/baseIF/IF-RL), unified panel (M1/M2/M3),
    m3v2 panel, native panel (nativeBase/N1..N4)
  - refold: refold5 (G0/M1/IF-RL), m2m3 refold, native refold, M3v2 own study
Outputs runs/report_unified/master_table.json + master_table.md
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
GUID = Path("/share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance")
UNI = GUID / "outputs/unified_valid100__g1_g2_g6__m1_m2_m3__g0_512_v1"
RU = ROOT / "runs/report_unified"

METHODS = ["G0", "M1", "M2", "M3", "M3v2", "baseIF", "IF-RL_R1", "IF-RL_R2",
           "nativeBase", "N1", "N2", "N3", "N4"]


def classifier_stats():
    out = {}
    # arm3: base IF / RL R1 / R2 / G0
    per = {}
    for line in open(ROOT / "runs/round1_rl_split/arm3_scores.jsonl"):
        r = json.loads(line)
        per.setdefault(r["arm"], []).append(r)
    for arm, key in [("base", "baseIF"), ("rl", "IF-RL_R1"), ("r2s1", "IF-RL_R2"), ("g0", "G0")]:
        rows = per.get(arm, [])
        if rows:
            out[key] = {
                "cdr_camelid_margin": st.mean(r["cdr_camelid_margin"] for r in rows),
                "nativeness": st.mean(r["camelid_native_likeness_score"] for r in rows),
                "discordance": st.mean(r["background_discordance_jsd"] for r in rows),
            }
    # unified panel M1/M2/M3
    per = {}
    for line in open(UNI / "classifier_scores.csv"):
        r = csv.DictReader  # noop
    for r in csv.DictReader((UNI / "classifier_scores.csv").open()):
        per.setdefault(r["method"], []).append(r)
    for m in ("M1", "M2", "M3"):
        rows = per.get(m, [])
        if rows:
            out[m] = {
                "cdr_camelid_margin": st.mean(float(r["cdr_camelid_margin"]) for r in rows),
                "nativeness": st.mean(float(r["camelid_native_likeness_score"]) for r in rows),
                "discordance": st.mean(float(r["background_discordance_jsd"]) for r in rows),
            }
    # M3v2
    p = RU / "m3v2_classifier_summary.json"
    if p.is_file():
        d = json.loads(p.read_text())
        out["M3v2"] = {
            "cdr_camelid_margin": d["cdr_camelid_margin"],
            "nativeness": d["camelid_native_likeness_score"],
            "discordance": d["background_discordance_jsd"],
        }
    # native panel
    p = ROOT / "runs/native_pool/native_panel_valid100_summary.json"
    if p.is_file():
        d = json.loads(p.read_text())
        for arm, key in [("base", "nativeBase"), ("n1", "N1"), ("n2", "N2"),
                         ("n3", "N3"), ("n4", "N4")]:
            out[key] = {
                "cdr_camelid_margin": d[arm]["cdr_camelid_margin"],
                "nativeness": d[arm]["camelid_native_likeness_score"],
                "discordance": d[arm]["background_discordance_jsd"],
            }
    return out


def refold_stats():
    out = {}
    # refold5: arms base/rl/r2s1/g0/m1
    path = ROOT / "runs/round1_rl_split/refold5_results_per_sample.csv"
    man = ROOT / "runs/round1_rl_split/refold5_design/manifest.csv"
    if path.is_file() and man.is_file():
        m = {r["sample_id"]: r["arm"] for r in csv.DictReader(man.open())}
        per = {}
        for r in csv.DictReader(path.open()):
            per.setdefault(m[r["sample_id"]], []).append(r)
        for arm, key in [("g0", "G0"), ("m1", "M1"), ("base", "baseIF"),
                         ("rl", "IF-RL_R1"), ("r2s1", "IF-RL_R2")]:
            rows = per.get(arm, [])
            if rows:
                out[key] = st.mean(float(r["cdr_rmsd"]) for r in rows)
    # M2/M3
    path = RU / "refold_m2m3_design/refold_cif"
    res = list(RU.glob("refold_m2m3_results*_per_sample.csv"))
    for p in res:
        man2 = RU / "refold_m2m3_design/manifest.csv"
        m = {r["sample_id"]: r["arm"] for r in csv.DictReader(man2.open())}
        per = {}
        for r in csv.DictReader(p.open()):
            per.setdefault(m[r["sample_id"]], []).append(r)
        for arm in ("M2", "M3"):
            rows = per.get(arm, [])
            if rows:
                out[arm] = st.mean(float(r["cdr_rmsd"]) for r in rows)
    # native
    path = ROOT / "runs/native_pool/native_refold_results_per_sample.csv"
    man = ROOT / "runs/native_pool/native_refold_design/manifest.csv"
    if path.is_file() and man.is_file():
        m = {r["sample_id"]: r["arm"] for r in csv.DictReader(man.open())}
        per = {}
        for r in csv.DictReader(path.open()):
            per.setdefault(m[r["sample_id"]], []).append(r)
        for arm, key in [("base", "nativeBase"), ("n1", "N1"), ("n2", "N2"),
                         ("n3", "N3"), ("n4", "N4")]:
            rows = per.get(arm, [])
            if rows:
                out[key] = st.mean(float(r["cdr_rmsd"]) for r in rows)
    # M3v2 own study
    p = GUID / "outputs/cdr_all/refold/m3v2_rmsd_per_sample.csv"
    if p.is_file():
        rows = list(csv.DictReader(p.open()))
        out["M3v2"] = st.mean(float(r["cdr_rmsd"]) for r in rows)
    return out


def main() -> None:
    if (RU / "unified_summary.json").is_file():
        seq = json.loads((RU / "unified_summary.json").read_text())
    else:
        import collections as _c
        per = _c.defaultdict(list)
        for line in (RU / "unified_scores.jsonl").open():
            r = json.loads(line)
            per[r["method"]].append(r)
        seq = {m: {"n": len(v),
                   "cdr_pll": st.mean(x["cdr_pll"] for x in v),
                   "full_pll": st.mean(x["full_pll"] for x in v),
                   "mpnn_nll": st.mean(x["mpnn_nll"] for x in v),
                   "recovery": st.mean(x["recovery"] for x in v)}
               for m, v in per.items()}
    # native sequence metrics may still be scoring; merge incremental when summary absent
    native_pending = [m for m in METHODS if m not in seq]
    cls = classifier_stats()
    ref = refold_stats()

    table = {}
    for m in METHODS:
        row = {"method": m}
        if m in seq:
            row.update({k: seq[m][k] for k in ("cdr_pll", "full_pll", "mpnn_nll", "recovery")})
        if m in cls:
            row.update(cls[m])
        if m in ref:
            row["refold_cdr_rmsd"] = ref[m]
        table[m] = row

    (RU / "master_table.json").write_text(json.dumps(table, indent=1))

    # markdown
    cols = [("cdr_pll", "CDR PLL↑"), ("cdr_camelid_margin", "CDR margin↑"),
            ("recovery", "recovery↑"), ("mpnn_nll", "MPNN NLL↓"),
            ("refold_cdr_rmsd", "refold RMSD↓"), ("nativeness", "nativeness"),
            ("discordance", "discordance↓")]
    lines = ["| method | " + " | ".join(t for _, t in cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for m in METHODS:
        row = table[m]
        cells = []
        for k, _ in cols:
            v = row.get(k)
            if v is None:
                cells.append("—")
            elif k in ("nativeness",):
                cells.append(f"{v:.3f}")
            elif k in ("discordance",):
                cells.append(f"{v:.3f}")
            else:
                cells.append(f"{v:.2f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    (RU / "master_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
