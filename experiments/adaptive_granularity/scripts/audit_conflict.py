#!/usr/bin/env python
"""Phase A1: conflict / reliability audit + Gate A (task book §8-§15).

CPU only.  Produces pair_region_audit.csv, pair_conflict_mass.csv,
audit_summary.json, figures/ and docs/adaptive_granularity/AG_AUDIT.md.
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

import _common as C  # noqa: E402
from vhh_rl.credit.adaptive_granularity import (  # noqa: E402
    REGION_STABLE_POSITIVE, RHO, SIGN_FLIP, STABLE_NEGATIVE, STABLE_POSITIVE,
    TOL, WEAK, adaptive_weights, classify_region, classify_residue,
)

OUT = C.AUDIT_DIR
FIGS = OUT / "figures"
DOC = C.ROOT / "docs/adaptive_granularity/AG_AUDIT.md"
GATE_A_MASS = 0.08
GATE_A_REGION_SHARE = 0.20
GATE_A_PAIR_SHARE = 0.20


def _median(values: list[float]) -> float | None:
    import statistics as st
    return st.median(values) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    pairs = C.load_pairs()
    residue = C.load_residue_rows()
    regions = C.load_region_rows()
    current = C.load_current_weights()

    region_rows_out, pair_rows_out = [], []
    mode_counts = defaultdict(int)
    rescue_rows = []
    for pair in pairs:
        pid = C.pair_id(pair)
        rows = residue.get(pid)
        if not rows:
            continue
        rregions = regions.get(pid, {})
        entry = current["pairs"].get(pid, {})
        cf = {int(k): float(v) for k, v in entry.get("cf", {}).items()}
        cf_total = sum(cf.values()) or 1.0
        class_of = {}
        mass = defaultdict(float)
        for r in rows:
            cls = classify_residue(r.get("c_drop"), r.get("c_gain"), TOL)
            class_of[r["position"]] = cls
            mass[cls] += cf.get(r["position"], 0.0) / cf_total
        pair_rows_out.append({
            "pair_id": pid, "case_id": pair["case_id"],
            "n_diff": len(rows),
            "mass_stable_positive": mass.get(STABLE_POSITIVE, 0.0),
            "mass_stable_negative": mass.get(STABLE_NEGATIVE, 0.0),
            "mass_sign_flip": mass.get(SIGN_FLIP, 0.0),
            "mass_weak": mass.get(WEAK, 0.0),
            "conflict_mass": mass.get(STABLE_NEGATIVE, 0.0) + mass.get(SIGN_FLIP, 0.0),
            "uncertain_mass": (mass.get(STABLE_NEGATIVE, 0.0)
                               + mass.get(SIGN_FLIP, 0.0) + mass.get(WEAK, 0.0)),
            "cf_fallback": int(bool(entry.get("fallback"))),
        })

        positions_by_region = defaultdict(list)
        for r in rows:
            positions_by_region[r["region"]].append(r["position"])
        for region, positions in sorted(positions_by_region.items()):
            subset = [r for r in rows if r["region"] == region]
            counts = defaultdict(int)
            for r in subset:
                counts[class_of[r["position"]]] += 1
            n_diff = len(subset)
            n_pos = counts.get(STABLE_POSITIVE, 0)
            rho_positive = n_pos / n_diff if n_diff else 0.0
            rho_conflict = (counts.get(STABLE_NEGATIVE, 0)
                            + counts.get(SIGN_FLIP, 0)) / n_diff if n_diff else 0.0
            rrow = rregions.get(region, {})
            region_class = classify_region(rrow.get("G_drop"), rrow.get("G_gain"), TOL)
            if rho_positive >= RHO and n_pos > 0:
                rescue = "residue_candidate"
            elif region_class == REGION_STABLE_POSITIVE:
                rescue = "coarse_rescue"
            else:
                rescue = "abstain"
            mode_counts[rescue] += 1
            pos_cons = [r.get("c_cons") or 0.0 for r in subset
                        if class_of[r["position"]] == STABLE_POSITIVE]
            region_rows_out.append({
                "pair_id": pid, "case_id": pair["case_id"], "region": region,
                "n_diff": n_diff,
                "n_stable_positive": n_pos,
                "n_stable_negative": counts.get(STABLE_NEGATIVE, 0),
                "n_sign_flip": counts.get(SIGN_FLIP, 0),
                "n_weak": counts.get(WEAK, 0),
                "rho_positive": rho_positive,
                "rho_conflict": rho_conflict,
                "sum_c_cons": sum((r.get("c_cons") or 0.0) for r in subset),
                "mean_c_cons_positive": (sum(pos_cons) / len(pos_cons)) if pos_cons else None,
                "G_drop": rrow.get("G_drop"), "G_gain": rrow.get("G_gain"),
                "G_cons": rrow.get("G_cons"), "G_avg": rrow.get("G_avg"),
                "region_class": region_class,
                "epistasis_avg": rrow.get("epistasis_avg"),
                "epistasis_cons": rrow.get("epistasis_cons"),
                "rescue_class": rescue,
            })
            rescue_rows.append({**region_rows_out[-1],
                                "cf_region_mass": sum(cf.get(p, 0.0) / cf_total
                                                      for p in positions)})

    with (OUT / "pair_region_audit.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(region_rows_out[0]))
        writer.writeheader()
        writer.writerows(region_rows_out)
    with (OUT / "pair_conflict_mass.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(pair_rows_out[0]))
        writer.writeheader()
        writer.writerows(pair_rows_out)

    conflict = [r["conflict_mass"] for r in pair_rows_out]
    uncertain = [r["uncertain_mass"] for r in pair_rows_out]
    n_regions = len(region_rows_out)
    n_rescue = mode_counts.get("coarse_rescue", 0)
    pairs_with_rescue = len({r["pair_id"] for r in rescue_rows
                             if r["rescue_class"] == "coarse_rescue"})
    gate_a = {
        "A_median_conflict_mass": {
            "value": _median(conflict), "threshold": GATE_A_MASS,
            "pass": (_median(conflict) or 0.0) >= GATE_A_MASS},
        "B_coarse_rescue_region_share": {
            "value": n_rescue / n_regions if n_regions else 0.0,
            "threshold": GATE_A_REGION_SHARE,
            "pass": (n_rescue / n_regions if n_regions else 0.0) >= GATE_A_REGION_SHARE},
        "C_pairs_with_coarse_rescue_share": {
            "value": pairs_with_rescue / len(pair_rows_out) if pair_rows_out else 0.0,
            "threshold": GATE_A_PAIR_SHARE,
            "pass": (pairs_with_rescue / len(pair_rows_out) if pair_rows_out else 0.0)
                    >= GATE_A_PAIR_SHARE},
    }
    gate_pass = any(v["pass"] for v in gate_a.values())

    adaptive_modes = defaultdict(int)
    for pair in pairs:
        pid = C.pair_id(pair)
        rows = residue.get(pid)
        if not rows:
            continue
        out = adaptive_weights(rows, regions.get(pid, {}), rho=RHO, tol=TOL)
        for mode in out.modes.values():
            adaptive_modes[mode] += 1

    summary = {
        "n_pairs": len(pair_rows_out),
        "n_regions": n_regions,
        "n_cases": len({r["case_id"] for r in pair_rows_out}),
        "residue_class_fraction": {
            cls: (sum(r[key] for r in region_rows_out)
                  / max(1, sum(r["n_diff"] for r in region_rows_out)))
            for cls, key in (("stable_positive", "n_stable_positive"),
                             ("stable_negative", "n_stable_negative"),
                             ("sign_flip", "n_sign_flip"), ("weak", "n_weak"))},
        "region_class_counts": _counts(region_rows_out, "region_class"),
        "rescue_class_counts": dict(mode_counts),
        "adaptive_mode_counts": dict(adaptive_modes),
        "conflict_mass": {
            "median": _median(conflict), "mean": (sum(conflict) / len(conflict)
                                                  if conflict else None),
            "p25": _quantile(conflict, 0.25), "p75": _quantile(conflict, 0.75),
            "fraction_above_0.10": (sum(1 for v in conflict if v > 0.10)
                                    / len(conflict)) if conflict else None,
        },
        "uncertain_mass": {
            "median": _median(uncertain),
            "fraction_above_0.25": (sum(1 for v in uncertain if v > 0.25)
                                    / len(uncertain)) if uncertain else None,
        },
        "gate_a": gate_a,
        "gate_a_pass": gate_pass,
        "rho": RHO, "tol": TOL,
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=1))

    _figures(region_rows_out, pair_rows_out, adaptive_modes)
    _write_doc(summary, gate_a, gate_pass)
    print(json.dumps({k: summary[k] for k in
                      ("n_pairs", "n_regions", "conflict_mass", "rescue_class_counts",
                       "adaptive_mode_counts", "gate_a_pass")}, indent=1))


def _counts(rows: list[dict], key: str) -> dict:
    out = defaultdict(int)
    for row in rows:
        out[row[key]] += 1
    return dict(out)


def _boxplot(ax, data, names) -> None:
    """Portable boxplot: matplotlib >= 3.9 renamed ``labels`` to ``tick_labels``."""
    try:
        ax.boxplot(data, tick_labels=names)
    except TypeError:
        ax.boxplot(data, labels=names)


def _figures(region_rows, pair_rows, adaptive_modes) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # fig1: current-CF conflict weight mass
    values = [r["conflict_mass"] for r in pair_rows]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(values, bins=25, color="#4c72b0")
    ax.axvline(_median(values), color="k", ls="--",
               label=f"median {_median(values):.3f}")
    ax.axvline(0.08, color="r", ls=":", label="Gate A 0.08")
    ax.set_xlabel("current-CF conflict mass per pair")
    ax.set_ylabel("#pairs")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(C.AUDIT_DIR / "figures/fig1_conflict_mass.png", dpi=150)
    plt.close(fig)

    # fig2: rho_positive by CDR
    by_region = defaultdict(list)
    for row in region_rows:
        by_region[row["region"]].append(row["rho_positive"])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    names = sorted(by_region)
    _boxplot(ax, [by_region[r] for r in names], names)
    ax.axhline(0.70, color="r", ls=":", label="rho=0.70")
    ax.set_ylabel("rho_positive")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(C.AUDIT_DIR / "figures/fig2_rho_by_region.png", dpi=150)
    plt.close(fig)

    # fig3: residue vs region reliability
    fig, ax = plt.subplots(figsize=(5, 3.2))
    xs = [r["rho_positive"] for r in region_rows]
    ys = [r["rho_conflict"] for r in region_rows]
    colors = {"stable_positive_region": "#55a868", "region_flip": "#c44e52",
              "stable_negative_region": "#8172b2", "region_weak": "#999999"}
    for cls in sorted(set(colors) & {r["region_class"] for r in region_rows}):
        idx = [i for i, r in enumerate(region_rows) if r["region_class"] == cls]
        ax.scatter([xs[i] for i in idx], [ys[i] for i in idx], s=8,
                   color=colors[cls], label=cls)
    ax.set_xlabel("rho_positive (residue reliability)")
    ax.set_ylabel("rho_conflict")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(C.AUDIT_DIR / "figures/fig3_residue_vs_region.png", dpi=150)
    plt.close(fig)

    # fig4: adaptive mode counts
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.bar(list(adaptive_modes), list(adaptive_modes.values()),
           color=["#55a868", "#4c72b0", "#c44e52"][:len(adaptive_modes)])
    ax.set_ylabel("#regions")
    fig.tight_layout()
    fig.savefig(C.AUDIT_DIR / "figures/fig4_adaptive_modes.png", dpi=150)
    plt.close(fig)

    # fig5: epistasis vs selected granularity
    fig, ax = plt.subplots(figsize=(5, 3.2))
    data = [[r["epistasis_cons"] for r in region_rows
             if r["rescue_class"].startswith(r_[:6]) and r["epistasis_cons"] is not None]
            for r_ in ("residue_candidate", "coarse_rescue", "abstain")]
    _boxplot(ax, data, ["residue", "region", "abstain"])
    ax.set_ylabel("epistasis_cons")
    fig.tight_layout()
    fig.savefig(C.AUDIT_DIR / "figures/fig5_epistasis_mode.png", dpi=150)
    plt.close(fig)


def _write_doc(summary: dict, gate_a: dict, gate_pass: bool) -> None:
    DOC.parent.mkdir(parents=True, exist_ok=True)
    def row(name: str, payload: dict) -> str:
        return (f"| {name} | {payload['value']:.4f} | {payload['threshold']} | "
                f"{'PASS' if payload['pass'] else 'fail'} |")
    table = "\n".join([
        row("A median current-CF conflict mass", gate_a["A_median_conflict_mass"]),
        row("B coarse-rescue region share", gate_a["B_coarse_rescue_region_share"]),
        row("C pairs with coarse-rescue share", gate_a["C_pairs_with_coarse_rescue_share"]),
    ])
    text = f"""# AG-CF-DPO Audit（task book §8-§15）

来源：`runs/next_stage/counterfactual/{{residue,region}}_credit.csv` +
`runs/next_stage/weights/residue_weights.json`（只读，未改动）。

## 1. 规模

- pairs {summary['n_pairs']}，regions {summary['n_regions']}，cases {summary['n_cases']}
- TOL={summary['tol']}，RHO={summary['rho']}（预注册，未调参）
- residue 类别比例：`{json.dumps(summary['residue_class_fraction'])}`
- region 类别计数：`{json.dumps(summary['region_class_counts'])}`

## 2. Current-CF conflict floor leakage（§12）

- conflict mass：median {summary['conflict_mass']['median']:.4f}，
  mean {summary['conflict_mass']['mean']:.4f}，
  P25 {summary['conflict_mass']['p25']:.4f}，P75 {summary['conflict_mass']['p75']:.4f}
- conflict mass > 0.10 的 pair 比例：{summary['conflict_mass']['fraction_above_0.10']:.3f}
- uncertain mass（含 weak）：median {summary['uncertain_mass']['median']:.4f}，
  > 0.25 比例 {summary['uncertain_mass']['fraction_above_0.25']:.3f}

## 3. Coarse-rescue coverage（§13）

- residue candidate regions：{summary['rescue_class_counts'].get('residue_candidate', 0)}
- **coarse rescue regions：{summary['rescue_class_counts'].get('coarse_rescue', 0)}**
- abstain regions：{summary['rescue_class_counts'].get('abstain', 0)}
- adaptive 预演 mode counts：`{json.dumps(summary['adaptive_mode_counts'])}`

## 4. Gate A（§14）

| condition | value | threshold | verdict |
|---|---:|---:|---|
{table}

**Gate A：{'PASS' if gate_pass else 'FAIL'}**（任意一条成立即通过；FAIL → 不训练）

## 5. Figures

- fig1 conflict weight mass
- fig2 rho_positive by CDR
- fig3 residue-vs-region reliability
- fig4 adaptive mode counts
- fig5 epistasis vs selected granularity
"""
    DOC.write_text(text)


if __name__ == "__main__":
    main()
