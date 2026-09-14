#!/usr/bin/env python
"""Paper-stage checkpoint selection (task book §54).

New multiseed runs: pick the best checkpoint on the 8 held-out cases with the
frozen rule (max reward, invalid <= base+1pp, FR mismatch == 0).
Primary-seed runs reuse their already-selected checkpoints (documented).
Writes runs/paper_stage/multiseed/checkpoint_selection.{json,md}
"""
from __future__ import annotations

import glob
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
POOL = ROOT / "runs/native_pool"
OUT = ROOT / "runs/paper_stage/multiseed"
OUT.mkdir(parents=True, exist_ok=True)

NEW_RUNS = {  # run_dir -> (arm, eval tag prefix)
    "n3_s43": "n3", "n3_s44": "n3",
    "shuffle_s43": "shuffle", "shuffle_s44": "shuffle",
    "cf_s43": "cf", "cf_s44": "cf",
}
PRIMARY = {
    "n3": {"summary": "eval_summary_n3_heldout_shard0.json", "selected_step": 350,
           "ckpt": "runs/native_n3/checkpoint_0350.pt",
           "valid100_tag": "n3_s0350", "seed": 20260913},
    "shuffle": {"summary": "eval_summary_full_b2_s0450_heldout_shard0.json", "selected_step": 450,
                "ckpt": "runs/next_stage/full_shuffle/checkpoint_0450.pt",
                "valid100_tag": "full_b2_s0450", "seed": 20260913},
    "cf": {"summary": "eval_summary_full_b3_s0500_heldout_shard0.json", "selected_step": 500,
           "ckpt": "runs/next_stage/full_cf/checkpoint_0500.pt",
           "valid100_tag": "full_b3_s0500", "seed": 20260913},
}


def summarize(summary_path: Path) -> dict:
    s = json.loads(summary_path.read_text())
    cases = s["cases"]
    n = sum(v["n_samples"] for v in cases.values())
    invalid = sum(v["n_invalid"] for v in cases.values())
    fr = sum(v["n_fr_mismatch"] for v in cases.values())
    scored = [v["reward_mean"] for v in cases.values() if v["reward_mean"] is not None]
    return {
        "reward_mean": st.mean(scored) if scored else None,
        "invalid_rate": invalid / max(n, 1),
        "fr_mismatch": fr, "n": n,
        "per_case": {k: v["reward_mean"] for k, v in cases.items()},
    }


def main() -> None:
    # base held-out from the frozen base pool
    base_cases = {}
    for d in sorted((POOL / "heldout").iterdir()):
        if not d.is_dir():
            continue
        rs = [json.loads(l)["reward_raw"] for l in (d / "metadata.jsonl").open()]
        rs = [r for r in rs if r is not None]
        if rs:
            base_cases[d.name] = st.mean(rs)
    base_reward = st.mean(base_cases.values())
    base_invalid = 0.0  # round-1 base held-out: 0 invalid / 0 FR mismatch
    print(f"base heldout reward={base_reward:+.3f}")

    selection = {"base_heldout_reward": base_reward, "runs": {}}
    for run, arm in sorted(NEW_RUNS.items()):
        rows = []
        for path in sorted(glob.glob(str(POOL / f"eval_summary_paper_{run}_s*_heldout_shard0.json"))):
            step = int(re.search(r"_s(\d+)_heldout", Path(path).name).group(1))
            info = summarize(Path(path))
            info["step"] = step
            rows.append(info)
        rows.sort(key=lambda r: r["step"])
        ok = [r for r in rows if r["reward_mean"] is not None
              and r["invalid_rate"] <= base_invalid + 0.01 and r["fr_mismatch"] == 0]
        best = max(ok, key=lambda r: r["reward_mean"]) if ok else None
        selection["runs"][run] = {
            "arm": arm, "rows": [{"step": r["step"], "reward_mean": r["reward_mean"],
                                  "invalid_rate": r["invalid_rate"], "fr_mismatch": r["fr_mismatch"]}
                                 for r in rows],
            "selected": best["step"] if best else None,
            "selected_reward": best["reward_mean"] if best else None,
            "ckpt": f"runs/paper_stage/{run}/checkpoint_{best['step']:04d}.pt" if best else None,
            "seed": 43 if run.endswith("43") else 44,
        }
        print(f"[{run}] " + " ".join(f"s{r['step']}:{r['reward_mean']:+.2f}" for r in rows
                                     if r["reward_mean"] is not None) +
              f" -> selected {best['step'] if best else None}")
    selection["primary"] = PRIMARY
    (OUT / "checkpoint_selection.json").write_text(json.dumps(selection, indent=1))

    md = "# Paper-stage checkpoint selection（held-out 8 cases, task book §54）\n\n"
    md += f"base held-out reward = {base_reward:+.3f}\n\n"
    md += "| run | arm | seed | selected step | held-out reward |\n|---|---|---|---|---|\n"
    for run, info in sorted(selection["runs"].items()):
        md += (f"| {run} | {info['arm']} | {info['seed']} | {info['selected']} | "
               f"{info['selected_reward']:+.3f} |\n")
    for arm, info in PRIMARY.items():
        md += f"| {arm}_primary | {arm} | 20260913 | {info['selected_step']} | (round-1 selected) |\n"
    (OUT / "checkpoint_selection.md").write_text(md)
    print(f"wrote {OUT / 'checkpoint_selection.md'}")


if __name__ == "__main__":
    main()
