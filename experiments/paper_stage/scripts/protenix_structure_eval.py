#!/usr/bin/env python
"""Paper-stage Step 4b: case-level CDR metrics for Protenix-v2 refolds.

For every Protenix prediction (candidate chain A) compare against the same
design backbone used by the Boltz-2 validation (self-consistency target),
compute CDR metrics, then case-level paired stats Base/N3/CF.

Writes results/paper_stage/structure_protenix_per_case.csv + stats.json/md
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
GUID = Path("/share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance")
sys.path.insert(0, str(GUID / "src"))
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "runs/paper_stage/structure_protenix"
DESIGN = ROOT / "runs/paper_stage/structure_boltz2"
RESULTS = ROOT / "results/paper_stage"
CACHE = GUID / "cache/protenix_native_structures/valid100"
METRICS = ["cdr_rmsd", "cdr1_rmsd", "cdr2_rmsd", "cdr3_rmsd", "fr_rmsd", "plddt"]


def cdr_groups(case_manifest: dict, cid: str) -> list[list[int]]:
    row = case_manifest[cid]
    from vhh_rl.data.cdr import cdr_range

    class _C:
        design_positions = tuple(row["design_positions"])

    return [list(cdr_range(_C(), k)) for k in ("cdr1", "cdr2", "cdr3")]


def main() -> None:
    from vhh_esmc_guidance.eval.candidate_structure_metrics import compute_structure_metrics

    rows = json.loads((OUT / "manifest.json").read_text())
    # valid100 case metadata: design positions for CDR groups
    case_manifest = {}
    for line in (ROOT / "runs/round1_rl_split/rl_manifest_valid100test.jsonl").open():
        r = json.loads(line)
        case_manifest[r["case_id"]] = {"design_positions": [int(p) for p in r["design_positions"]]}

    out_rows = []
    for r in rows:
        cid = r["case_id"]
        candidate = CACHE / cid / "structure.cif"
        idx = int(cid[2:5])
        target = DESIGN / f"p{idx:03d}b0.cif"
        if not candidate.is_file() or not target.is_file():
            out_rows.append({**r, "status": "MISSING"})
            continue
        try:
            metrics = compute_structure_metrics(
                candidate, target, cdr_groups(case_manifest, r["valid100_case"]),
                candidate_chain_id="A")
            out_rows.append({**r, **{m: metrics.get(m) for m in METRICS},
                             "status": metrics.get("status", "PASS")})
        except Exception as exc:  # noqa: BLE001
            out_rows.append({**r, "status": f"ERROR: {type(exc).__name__}: {exc}"})

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "structure_protenix_per_case.csv").open("w", newline="") as fh:
        fields = ["case_id", "arm", "valid100_case", "idx", "sequence", "status"] + METRICS
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    # case-level means per arm
    def case_means(metric: str) -> dict[tuple[str, str], float]:
        acc: dict[tuple[str, str], list[float]] = {}
        for r in out_rows:
            v = r.get(metric)
            if isinstance(v, (int, float)):
                acc.setdefault((r["valid100_case"], r["arm"]), []).append(float(v))
        return {k: st.mean(v) for k, v in acc.items()}

    cm = case_means("cdr_rmsd")
    arms = ["base", "n3", "cf"]
    cases = sorted({c for c, _a in cm})
    entry = {a: [cm[(c, a)] for c in cases if (c, a) in cm] for a in arms}
    stats = {"n_cases": len(cases)}
    for a in arms:
        stats[a] = {"cdr_rmsd_mean": st.mean(entry[a]) if entry[a] else None,
                    "n": len(entry[a])}
    for ref in ("base", "n3"):
        d = np.array([cm[(c, "cf")] - cm[(c, ref)] for c in cases
                      if (c, "cf") in cm and (c, ref) in cm])
        rng = np.random.default_rng(12345)
        boots = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
        try:
            from scipy.stats import wilcoxon
            p = float(wilcoxon(d, zero_method="wilcox").pvalue) if any(d != 0) else 1.0
        except Exception:
            p = None
        stats[f"cf_vs_{ref}"] = {
            "n": int(len(d)), "delta_mean": float(d.mean()),
            "wins_lower": int((d < 0).sum()), "losses": int((d > 0).sum()),
            "ci95_low": float(np.percentile(boots, 2.5)),
            "ci95_high": float(np.percentile(boots, 97.5)), "wilcoxon_p": p,
        }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    md = "# Protenix-v2 independent refold (50 cases x 2 seqs, case-level)\n\n"
    for a in arms:
        md += f"- {a}: CDR RMSD mean = {stats[a]['cdr_rmsd_mean']:.3f} (n={stats[a]['n']})\n"
    for ref in ("base", "n3"):
        v = stats[f"cf_vs_{ref}"]
        md += (f"- CF vs {ref}: Δ mean {v['delta_mean']:+.3f} Å, "
               f"CI [{v['ci95_low']:+.3f}, {v['ci95_high']:+.3f}], "
               f"W/L {v['wins_lower']}/{v['losses']}, p={v['wilcoxon_p']}\n")
    (OUT / "stats.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
