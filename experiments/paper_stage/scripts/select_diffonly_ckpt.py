#!/usr/bin/env python
"""Held-out checkpoint selection for Diff-only seeds 43/44 (task book §11).

Same frozen rule as the paper-stage multiseed selection: max held-out reward
with invalid_rate <= base + 1pp and FR mismatch == 0.  valid100 is never used
for selection.  Also writes the frozen valid100 task list for the two
selected checkpoints.
"""
from __future__ import annotations

import glob
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
POOL = ROOT / "runs/native_pool"
OUT = ROOT / "runs/paper_stage/diffonly_multiseed"
NEW_RUNS = ["diff_only_s43", "diff_only_s44"]
PRIMARY = {  # §8: reuse the existing primary artifact, do not retrain
    "run": "diff_only", "seed": 20260913, "selected_step": 500,
    "ckpt": "runs/paper_stage/diff_only/checkpoint_0500.pt",
    "valid100_tag": "paper_diff_only_s0500",
    "selection_source": "paper-stage ablation sweep (already published)",
}
SHARDS = 4


def summarize(path: Path) -> dict:
    s = json.loads(path.read_text())
    cases = s["cases"]
    n = sum(v["n_samples"] for v in cases.values())
    invalid = sum(v["n_invalid"] for v in cases.values())
    fr = sum(v["n_fr_mismatch"] for v in cases.values())
    scored = [v["reward_mean"] for v in cases.values() if v["reward_mean"] is not None]
    return {"reward_mean": st.mean(scored) if scored else None,
            "invalid_rate": invalid / max(n, 1), "fr_mismatch": fr, "n": n,
            "per_case": {k: v["reward_mean"] for k, v in cases.items()}}


def base_heldout() -> tuple[float, float]:
    cases = {}
    for d in sorted((POOL / "heldout").iterdir()):
        if not d.is_dir():
            continue
        rs = [json.loads(l)["reward_raw"] for l in (d / "metadata.jsonl").open()]
        rs = [r for r in rs if r is not None]
        if rs:
            cases[d.name] = st.mean(rs)
    return st.mean(cases.values()), 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base_reward, base_invalid = base_heldout()
    print(f"base heldout reward={base_reward:+.3f}")
    selection = {"base_heldout_reward": base_reward, "primary": PRIMARY, "runs": {}}
    for run in NEW_RUNS:
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
        entry = {
            "rows": [{k: r[k] for k in ("step", "reward_mean", "invalid_rate",
                                        "fr_mismatch")} for r in rows],
            "selected": best["step"] if best else None,
            "selected_reward": best["reward_mean"] if best else None,
            "ckpt": (f"runs/paper_stage/{run}/checkpoint_{best['step']:04d}.pt"
                     if best else None),
            "seed": int(run.rsplit("s", 1)[1]),
            "valid100_tag": (f"paper_{run}_s{best['step']:04d}" if best else None),
        }
        selection["runs"][run] = entry
        (ROOT / "runs/paper_stage" / run / "checkpoint_selection.json").write_text(
            json.dumps(entry, indent=1))
        print(f"{run}: selected step {entry['selected']} "
              f"reward={entry['selected_reward']}")
    (OUT / "checkpoint_selection.json").write_text(json.dumps(selection, indent=1))
    md = ["# Diff-only multiseed checkpoint selection", "",
          f"base heldout reward: {base_reward:+.3f}", ""]
    for run, entry in selection["runs"].items():
        md += [f"## {run} (seed {entry['seed']})", "",
               "| step | heldout reward | invalid rate | FR |", "|---:|---:|---:|---:|"]
        md += [f"| {r['step']} | {r['reward_mean']:+.3f} | {r['invalid_rate']:.4f} | "
               f"{r['fr_mismatch']} |" for r in entry["rows"]]
        md += ["", f"selected: step {entry['selected']} "
                   f"(reward {entry['selected_reward']:+.3f})", ""]
    (OUT / "checkpoint_selection.md").write_text("\n".join(md) + "\n")

    tasks = []
    for run in NEW_RUNS:
        entry = selection["runs"][run]
        if not entry["ckpt"]:
            raise SystemExit(f"{run}: no valid checkpoint selected")
        for shard in range(SHARDS):
            tasks.append(f"{entry['valid100_tag']}\t{entry['ckpt']}\t{shard}")
    tasks_path = ROOT / "runs/paper_stage/valid100_tasks_diffonly.tsv"
    tasks_path.write_text("\n".join(tasks) + "\n")
    print(f"wrote {tasks_path} ({len(tasks)} tasks)")
    print("submit with: PAPER_V100_TASKS=%s sbatch --array=0-%d%%4 %s"
          % (tasks_path, len(tasks) - 1,
             ROOT / "experiments/paper_stage/scripts/sbatch_paper_valid100.sh"))


if __name__ == "__main__":
    main()
