#!/usr/bin/env python
"""Diff-only multi-seed aggregation, statistics and decision (task book §12-§24).

Reuses the frozen valid100 protocol outputs and writes:
  results/paper_stage/diffonly_multiseed.csv
  results/paper_stage/diffonly_seed_level.csv
  results/paper_stage/diffonly_per_case_s<seed>.csv
  results/paper_stage/diffonly_stats.csv
  docs/paper_stage/P0_DIFFONLY_MULTISEED.md
and appends the re-audit section to docs/paper_stage/PAPER_STAGE_RESULTS.md.
Never modifies existing single-seed results.
"""
from __future__ import annotations

import csv
import glob
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "experiments/paper_stage/scripts"))
POOL = ROOT / "runs/native_pool"
V100 = ROOT / "runs/paper_stage/valid100"
RESULTS = ROOT / "results/paper_stage"
DOC = ROOT / "docs/paper_stage/P0_DIFFONLY_MULTISEED.md"
RESULTS_DOC = ROOT / "docs/paper_stage/PAPER_STAGE_RESULTS.md"

SEEDS = [20260913, 43, 44]
# tags reuse the frozen paper-stage valid100 artifacts (§8: no retraining of
# the primary diff-only run; CF/N3 tags come from the published multiseed runs)
TAGS = {
    20260913: {"n3": "n3_s0350", "diff": "paper_diff_only_s0500",
               "cf": "full_b3_s0500"},
    43: {"n3": "paper_n3_s43_s0500", "diff": None, "cf": "paper_cf_s43_s0400"},
    44: {"n3": "paper_n3_s44_s0500", "diff": None, "cf": "paper_cf_s44_s0450"},
}
SHARD_GLOB = "eval_summary_{tag}_valid100_shard*.json"


def base_case_means() -> dict[str, float]:
    out = {}
    for d in sorted((POOL / "valid100").iterdir()):
        if not d.is_dir():
            continue
        rs = [json.loads(l)["reward_raw"] for l in (d / "metadata.jsonl").open()]
        rs = [r for r in rs if r is not None]
        if rs:
            out[d.name] = st.mean(rs)
    return out


def paired_stats(delta: list[float]) -> dict:
    from aggregate_valid100_paper import paired
    return paired(delta)


def diff_selected_tag(seed: int) -> str:
    payload = json.loads(
        (ROOT / "runs/paper_stage/diffonly_multiseed/checkpoint_selection.json").read_text())
    return payload["runs"][f"diff_only_s{seed}"]["valid100_tag"]


def aggregate_tag(tag: str) -> dict | None:
    """Merge valid100 shard summaries into runs/paper_stage/valid100/<tag>.json."""
    cases: dict[str, list[float]] = {}
    n_invalid = n_total = 0
    files = sorted(glob.glob(str(POOL / SHARD_GLOB.format(tag=tag))))
    if not files:
        return None
    for path in files:
        summary = json.loads(Path(path).read_text())
        for cid, v in summary["cases"].items():
            n_invalid += v["n_invalid"]
            n_total += v["n_samples"]
            if v["reward_mean"] is not None:
                cases.setdefault(cid, []).append(v["reward_mean"])
    means = {c: st.mean(v) for c, v in cases.items()}
    existing = V100 / f"{tag}.json"
    if existing.is_file():
        old = json.loads(existing.read_text())
        if old.get("tag") == tag and len(old.get("per_case", {})) >= len(means):
            return old
    base = base_case_means()
    common = sorted(set(means) & set(base))
    stats = paired_stats([means[c] - base[c] for c in common])
    payload = {"tag": tag, "reward_mean": st.mean(means.values()),
               "per_case": means, "delta_vs_base": stats,
               "invalid_rate": n_invalid / max(1, n_total), "n": n_total}
    V100.mkdir(parents=True, exist_ok=True)
    (V100 / f"{tag}.json").write_text(json.dumps(payload, indent=1))
    return payload


def load_valid100(tag: str) -> dict:
    path = V100 / f"{tag}.json"
    payload = json.loads(path.read_text())
    if "per_case" not in payload:
        raise SystemExit(f"{path} lacks per_case")
    return payload


def main() -> None:
    for seed in (43, 44):
        tag = diff_selected_tag(seed)
        built = aggregate_tag(tag)
        if built is None:
            raise SystemExit(f"missing valid100 shards for {tag}; run paper-diffonly-valid100")
        TAGS[seed]["diff"] = tag
    base = base_case_means()
    arms = {seed: {arm: load_valid100(tag) for arm, tag in TAGS[seed].items()}
            for seed in SEEDS}

    RESULTS.mkdir(parents=True, exist_ok=True)
    seed_rows, per_case_files = [], {}
    for seed in SEEDS:
        a = arms[seed]
        common = sorted(set(base) & set(a["n3"]["per_case"])
                        & set(a["diff"]["per_case"]) & set(a["cf"]["per_case"]))
        row = {"seed": seed,
               "n3_delta": a["n3"]["reward_mean"] - st.mean(base[c] for c in common),
               "diff_only_delta": a["diff"]["reward_mean"] - st.mean(base[c] for c in common),
               "cf_delta": a["cf"]["reward_mean"] - st.mean(base[c] for c in common)}
        row["cf_minus_diff"] = a["cf"]["reward_mean"] - a["diff"]["reward_mean"]
        row["diff_minus_n3"] = a["diff"]["reward_mean"] - a["n3"]["reward_mean"]
        row["cf_minus_n3"] = a["cf"]["reward_mean"] - a["n3"]["reward_mean"]
        seed_rows.append(row)
        per_case_files[seed] = [{
            "case_id": c, "base": base[c],
            "n3": a["n3"]["per_case"].get(c), "diff_only": a["diff"]["per_case"].get(c),
            "cf": a["cf"]["per_case"].get(c),
            "cf_minus_diff": (a["cf"]["per_case"].get(c, float("nan"))
                              - a["diff"]["per_case"].get(c, float("nan"))),
            "diff_minus_n3": (a["diff"]["per_case"].get(c, float("nan"))
                              - a["n3"]["per_case"].get(c, float("nan"))),
        } for c in common]

    # per-seed paired statistics: CF vs Diff-only and Diff-only vs N3
    stats_rows = []
    for seed in SEEDS:
        a = arms[seed]
        common = sorted(set(a["diff"]["per_case"]) & set(a["cf"]["per_case"])
                        & set(a["n3"]["per_case"]))
        for name, left, right in (("cf_vs_diff", "cf", "diff"),
                                  ("diff_vs_n3", "diff", "n3")):
            stats = paired_stats([a[left]["per_case"][c] - a[right]["per_case"][c]
                                  for c in common])
            stats_rows.append({"seed": seed, "comparison": name, **stats})

    write_csv(RESULTS / "diffonly_seed_level.csv", seed_rows)
    write_csv(RESULTS / "diffonly_stats.csv", stats_rows)
    write_csv(RESULTS / "diffonly_multiseed.csv", _main_table(seed_rows))
    for seed, rows in per_case_files.items():
        write_csv(RESULTS / f"diffonly_per_case_s{seed}.csv", rows)

    doc = build_doc(seed_rows, stats_rows, arms)
    DOC.write_text(doc)
    _update_results_doc(doc)
    q1 = (st.mean(r["diff_only_delta"] for r in seed_rows),
          st.pstdev(r["diff_only_delta"] for r in seed_rows))
    d = [r["cf_minus_diff"] for r in seed_rows]
    print(json.dumps({
        "Q1_diff_only_mean_std": q1,
        "Q2_cf_minus_diff": d,
        "Q3_three_of_three": all(x > 0 for x in d),
        "Q4_decision": decide(d),
    }, indent=1))


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _main_table(seed_rows: list[dict]) -> list[dict]:
    keys = ("n3_delta", "diff_only_delta", "cf_delta", "cf_minus_diff",
            "diff_minus_n3")
    rows = [dict(r) for r in seed_rows]
    for label, fn in (("mean", st.mean), ("std", st.pstdev)):
        rows.append({"seed": label,
                     **{k: fn([r[k] for r in seed_rows]) for k in keys}})
    return rows


def decide(deltas: list[float]) -> dict:
    mean_d = st.mean(deltas)
    three_of_three = all(x > 0 for x in deltas)
    if not three_of_three or mean_d < 0.30:
        case = "A"
        text = ("credit weighting 没有稳定独立价值；当前稳定收益主要来自 "
                "support localization")
    elif mean_d < 0.75:
        case = "B"
        text = "credit weighting 有稳定但中等幅度价值"
    else:
        case = "C"
        text = "counterfactual credit weighting 是核心算法成分"
    return {"case": case, "text": text, "mean_cf_minus_diff": mean_d,
            "three_of_three_positive": three_of_three}


def build_doc(seed_rows: list[dict], stats_rows: list[dict], arms: dict) -> str:
    d = [r["cf_minus_diff"] for r in seed_rows]
    decision = decide(d)
    sel = json.loads((ROOT / "runs/paper_stage/diffonly_multiseed/"
                      "checkpoint_selection.json").read_text())
    primary_sha = _sha256(ROOT / sel["primary"]["ckpt"])
    lines = [
        "# Diff-only Multi-seed Verification", "",
        "## 1. Question", "",
        "CF-DPO 的稳定收益来自 differing-residue support localization，"
        "还是来自 counterfactual credit weighting 本身？", "",
        "## 2. Frozen Protocol", "",
        "- 复用 paper-stage frozen protocol（BoltzGen `a3149cf`，beta=10，lr=1e-5，"
        "500 updates，24 train / 8 heldout / frozen valid100，8 samples/case）",
        "- primary diff-only 直接复用既有 artifact，未重训："
        f"`{sel['primary']['ckpt']}` sha256 `{primary_sha[:16]}…`，seed 20260913",
        "- 新增 diff-only seed 43/44，参数与既有 CF/N3 multiseed 完全一致", "",
        "## 3. Correctness Audit", "",
        "- `runs/paper_stage/diffonly_audit/audit_summary.json`："
        "192/192 pairs 通过；20/20 抽样通过",
        "- 每个 pair：support == differing positions、active 权重唯一、"
        "w = 1/n_diff、sum = 1、无 same-residue/非 design 位置权重", "",
        "## 4. Training Seeds", "",
        "| seed | run | selected step | heldout reward | valid100 tag |", "|---:|---|---:|---:|---|",
    ]
    for seed in SEEDS:
        if seed == 20260913:
            lines.append(f"| {seed} | diff_only (primary) | "
                         f"{sel['primary']['selected_step']} | - | "
                         f"{sel['primary']['valid100_tag']} |")
        else:
            entry = sel["runs"][f"diff_only_s{seed}"]
            lines.append(f"| {seed} | diff_only_s{seed} | {entry['selected']} | "
                         f"{entry['selected_reward']:+.3f} | {entry['valid100_tag']} |")
    lines += ["", "## 5. Held-out Checkpoint Selection", "",
              "选择规则：8 held-out cases 上 reward 最高，invalid ≤ base+1pp，FR=0；"
              "未使用 valid100 选择（见 `runs/paper_stage/diffonly_multiseed/"
              "checkpoint_selection.md`）。", "",
              "## 6. Valid100 Results", "",
              "| seed | N3 Δ | Diff-only Δ | CF Δ | CF−Diff | Diff−N3 |",
              "|---:|---:|---:|---:|---:|---:|"]
    for r in seed_rows:
        lines.append(f"| {r['seed']} | {r['n3_delta']:+.3f} | {r['diff_only_delta']:+.3f} | "
                     f"{r['cf_delta']:+.3f} | {r['cf_minus_diff']:+.3f} | "
                     f"{r['diff_minus_n3']:+.3f} |")
    for label, fn in (("mean", st.mean), ("std", st.pstdev)):
        lines.append(f"| {label} | {fn([r['n3_delta'] for r in seed_rows]):.3f} | "
                     f"{fn([r['diff_only_delta'] for r in seed_rows]):.3f} | "
                     f"{fn([r['cf_delta'] for r in seed_rows]):.3f} | "
                     f"{fn([r['cf_minus_diff'] for r in seed_rows]):.3f} | "
                     f"{fn([r['diff_minus_n3'] for r in seed_rows]):.3f} |")
    lines += ["", "## 7. Seed-level Comparison", "",
              "| seed | comparison | mean Δ | median Δ | W/T/L | Wilcoxon p | "
              "bootstrap 95% CI |", "|---:|---|---:|---:|---|---:|---|"]
    for row in stats_rows:
        ci = (f"[{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}]"
              if row.get("ci95_low") is not None else "-")
        p = row.get("wilcoxon_p")
        lines.append(f"| {row['seed']} | {row['comparison']} | {row['delta_mean']:+.3f} | "
                     f"{row['delta_median']:+.3f} | {row['wins']}/{row['ties']}/"
                     f"{row['losses']} | {'-' if p is None else f'{p:.2e}'} | {ci} |")
    lines += ["", "D_s = CF_s − DiffOnly_s："
              + ", ".join(f"D_{r['seed']} = {r['cf_minus_diff']:+.3f}" for r in seed_rows)
              + f"；mean(D) = {st.mean(d):+.3f}，std(D) = {st.pstdev(d):.3f}，"
                f"min(D) = {min(d):+.3f}，3/3 positive = {all(x > 0 for x in d)}",
              "", "## 8. Case-level Paired Statistics", "",
              "每 seed 的分 case 文件：`results/paper_stage/diffonly_per_case_s*.csv`"
              "（100 cases 配对；Wilcoxon + 10000 次 paired bootstrap，seed=12345）。", "",
              "## 9. Interpretation", "",
              f"- mean(CF−Diff) = {st.mean(d):+.3f}，3/3 seeds CF > Diff-only = "
              f"{all(x > 0 for x in d)}",
              "- 所有 seed 的 valid100 invalid/FR 见 `runs/paper_stage/valid100/*.json`；"
              "训练 sanity：non-finite=0、FR=0（`runs/paper_stage/diff_only_s*/"
              "train_metrics.jsonl`）",
              "", "## 10. Decision", "",
              f"**Case {decision['case']}**：{decision['text']}。", "",
              "## 11. Next Step", "",
              "- 若 Case A：停止 credit refinement（不再 sign/region/adaptive/Shapley）",
              "- 若 Case B/C：下一步只允许固定 support 下的 shrinkage curve "
              "η ∈ {0, 0.25, 0.5, 0.75, 1.0}（本任务不执行）",
              "- 完成后 STOP，等待人工审阅", ""]
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    import hashlib
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_results_doc(doc: str) -> None:
    marker = "## Diff-only Multi-seed Re-audit"
    section = marker + "\n\n" + doc.split("# Diff-only Multi-seed Verification", 1)[1]
    section = section.replace("\n## ", "\n### ")
    text = RESULTS_DOC.read_text()
    if marker in text:
        head = text.split(marker, 1)[0].rstrip()
        RESULTS_DOC.write_text(head + "\n\n" + section.strip() + "\n")
    else:
        RESULTS_DOC.write_text(text.rstrip() + "\n\n" + section.strip() + "\n")


if __name__ == "__main__":
    main()
