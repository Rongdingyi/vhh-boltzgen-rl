#!/usr/bin/env python
"""Paper-stage synthesis: aggregate all results into results/paper_stage/*.csv
and print the paper main tables (task book §50/§69).

Gracefully skips pieces that are not ready yet.
"""
from __future__ import annotations

import csv
import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
PS = ROOT / "runs/paper_stage"
RESULTS = ROOT / "results/paper_stage"
RESULTS.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def multiseed_rows() -> list[dict]:
    sel = read_json(PS / "multiseed/checkpoint_selection.json")
    if not sel:
        return []
    rows = []
    runs = dict(sel["runs"])
    for arm, info in sel.get("primary", {}).items():
        runs[f"{arm}_primary"] = {"arm": arm, "seed": info["seed"],
                                  "selected": info["selected_step"],
                                  "selected_reward": None, "ckpt": info["ckpt"],
                                  "valid100_tag": info["valid100_tag"]}
    for run, info in sorted(runs.items()):
        info = {**info, "run": run}
        v100 = read_json(PS / f"valid100/{_v100_tag(info)}.json")
        if v100 is None:
            v100 = read_json(PS / f"valid100/{_v100_tag({**info, 'valid100_tag': info.get('valid100_tag', '')})}.json")
        row = {
            "run": run, "arm": info["arm"], "seed": info.get("seed"),
            "selected_step": info.get("selected"),
            "heldout_reward": info.get("selected_reward"),
        }
        if v100:
            d = v100.get("delta_vs_base", {})
            row.update({
                "valid100_reward": v100.get("reward_mean"),
                "valid100_delta": d.get("delta_mean"),
                "wins": d.get("wins"), "losses": d.get("losses"),
                "wilcoxon_p": d.get("wilcoxon_p"),
                "invalid_rate": v100.get("invalid_rate"),
            })
        rows.append(row)
    return rows


def _v100_tag(info: dict) -> str:
    if "valid100_tag" in info:
        return info["valid100_tag"]
    run = info.get("run")
    step = info.get("selected")
    if run and step:
        return f"paper_{run}_s{int(step):04d}"
    return f"paper_{info.get('arm', '')}_{info.get('seed', '')}"


def ablation_rows() -> list[dict]:
    rows = []
    for tag in ("diff_only", "drop_only", "gain_only", "cf", "shuffle", "n3",
                "0.25", "0.50", "0.75"):
        for path in glob.glob(str(PS / f"valid100/paper_{tag}*.json")):
            v = read_json(Path(path))
            d = v.get("delta_vs_base", {})
            rows.append({
                "arm": tag, "reward_mean": v.get("reward_mean"),
                "delta_mean": d.get("delta_mean"), "wins": d.get("wins"),
                "n": d.get("n"), "wilcoxon_p": d.get("wilcoxon_p"),
                "invalid_rate": v.get("invalid_rate"),
            })
    return rows


def structure_rows() -> list[dict]:
    stats = read_json(PS / "structure_boltz2/stats.json")
    rows = []
    if stats:
        for metric in ("cdr_rmsd", "cdr3_rmsd", "fr_rmsd", "plddt_cdr"):
            e = stats["metrics"][metric]
            for arm in ("base", "n3", "shuffle", "cf"):
                rows.append({"refolder": "boltz2", "metric": metric, "arm": arm,
                             "case_mean": e[arm]["mean"]})
            for ref in ("base", "n3", "shuffle"):
                v = e[f"cf_vs_{ref}"]
                rows.append({"refolder": "boltz2", "metric": metric,
                             "arm": f"cf_vs_{ref}", "case_mean": v["delta_mean"],
                             "ci95_low": v["ci95_low"], "ci95_high": v["ci95_high"],
                             "wilcoxon_p": v["wilcoxon_p"]})
    return rows


def query_budget_rows() -> list[dict]:
    weights = read_json(PS / "weights/query_budget_weights.json")
    rows = []
    if not weights:
        return rows
    budgets = weights["budgets"]
    tag_map = {"0.25": "paper_qb_025_s0450", "0.50": "paper_qb_050_s0500",
               "0.75": "paper_qb_075_s0500", "1.00": "full_b3_s0500"}
    for tag, info in budgets.items():
        v = read_json(PS / f"valid100/{tag_map.get(tag, '')}.json")
        row = {"budget": tag, **info}
        if v:
            d = v.get("delta_vs_base", {})
            row.update({"valid100_reward": v.get("reward_mean"),
                        "delta_mean": d.get("delta_mean"), "wins": d.get("wins"),
                        "n": d.get("n"), "wilcoxon_p": d.get("wilcoxon_p")})
        rows.append(row)
    return rows


def second_reward_rows() -> list[dict]:
    summary = read_json(PS / "second_reward/heldout_summary.json")
    rows = []
    if not summary:
        return rows
    for arm, info in summary.get("arms", {}).items():
        rows.append({"arm": arm, **info})
    if "cf_vs_uniform" in summary:
        rows.append({"arm": "cf_vs_uniform", **summary["cf_vs_uniform"]})
    return rows


def main() -> None:
    ms = multiseed_rows()
    write_csv(RESULTS / "multiseed_summary.csv", ms)

    # seed-level mean +- std per arm (valid100 delta)
    per_arm: dict[str, list[float]] = {}
    for r in ms:
        if r.get("valid100_delta") is not None:
            per_arm.setdefault(r["arm"], []).append(float(r["valid100_delta"]))
    seed_rows = []
    for arm, vals in sorted(per_arm.items()):
        seed_rows.append({"arm": arm, "n_seeds": len(vals),
                          "delta_mean": st.mean(vals),
                          "delta_std": st.stdev(vals) if len(vals) > 1 else 0.0,
                          "delta_min": min(vals), "delta_max": max(vals)})
    write_csv(RESULTS / "multiseed_seed_level.csv", seed_rows)

    write_csv(RESULTS / "ablation_summary.csv", ablation_rows())
    write_csv(RESULTS / "structure_validation.csv", structure_rows())
    write_csv(RESULTS / "query_budget.csv", query_budget_rows())
    write_csv(RESULTS / "second_reward_summary.csv", second_reward_rows())

    # main paper table
    main_rows = []
    for arm, vals in sorted(per_arm.items()):
        main_rows.append({"method": arm, "delta_reward_mean": st.mean(vals),
                          "delta_reward_std": st.stdev(vals) if len(vals) > 1 else 0.0,
                          "n_seeds": len(vals)})
    if main_rows:
        write_csv(RESULTS / "paper_main_table.csv", main_rows)
    print(json.dumps({"multiseed_rows": len(ms), "seed_level": seed_rows}, indent=1, default=str))


if __name__ == "__main__":
    main()
