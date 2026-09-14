#!/usr/bin/env python
"""Phase C checkpoint selection on held-out (task book §40).

Criteria: highest held-out reward among checkpoints with
  invalid_rate <= base_invalid_rate + 1pp  AND  FR mismatch == 0.
Never looks at valid100.
Outputs: runs/next_stage/checkpoint_selection.json + .md
"""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
POOL = ROOT / "runs/native_pool"
OUT = ROOT / "runs/next_stage"
ARMS = {"full_b1": "full_random_sparse", "full_b2": "full_shuffle", "full_b3": "full_cf"}


def case_stats(pool_dir: Path) -> dict:
    rewards, invalid, fr, n = [], 0, 0, 0
    for meta in sorted(pool_dir.glob("*/metadata.jsonl")):
        for line in meta.open():
            r = json.loads(line)
            n += 1
            invalid += int(bool(r["contains_UNK"]))
            fr += int(int(r["FR_mismatch_count"]) > 0)
            if r["reward_raw"] is not None:
                rewards.append(float(r["reward_raw"]))
    return {
        "n": n, "n_scored": len(rewards),
        "reward_mean": st.mean(rewards) if rewards else None,
        "invalid_rate": invalid / max(n, 1),
        "fr_mismatch_rate": fr / max(n, 1),
    }


def main() -> None:
    base = case_stats(POOL / "heldout")
    print(f"base heldout: reward={base['reward_mean']:+.3f} "
          f"invalid={base['invalid_rate']:.3f} fr={base['fr_mismatch_rate']:.3f}")
    tables = {}
    selected = {}
    for tag, arm_dir in ARMS.items():
        rows = []
        for summary_path in sorted(glob.glob(str(POOL / f"eval_summary_{tag}_s*_heldout_shard0.json"))):
            summary = json.loads(Path(summary_path).read_text())
            cases = summary["cases"]
            n = sum(v["n_samples"] for v in cases.values())
            invalid = sum(v["n_invalid"] for v in cases.values())
            fr = sum(v["n_fr_mismatch"] for v in cases.values())
            scored = [v["reward_mean"] for v in cases.values() if v["reward_mean"] is not None]
            import re
            step = int(re.search(r"_s(\d+)_", Path(summary_path).name).group(1))
            rows.append({
                "step": step,
                "reward_mean": st.mean(scored) if scored else None,
                "invalid_rate": invalid / max(n, 1),
                "fr_mismatch": fr, "n": n,
                "path": str(Path(arm_dir) / f"checkpoint_{step:04d}.pt"),
            })
        rows.sort(key=lambda r: r["step"])
        ok = [r for r in rows
              if r["reward_mean"] is not None
              and r["invalid_rate"] <= base["invalid_rate"] + 0.01
              and r["fr_mismatch"] == 0]
        best = max(ok, key=lambda r: r["reward_mean"]) if ok else None
        tables[tag] = rows
        selected[tag] = best
        print(f"[{tag}] " + " ".join(
            f"s{r['step']}:{r['reward_mean']:+.2f}" if r["reward_mean"] is not None else f"s{r['step']}:n/a"
            for r in rows))
        print(f"[{tag}] selected: {best['step'] if best else None} "
              f"reward={best['reward_mean'] if best else None}")

    payload = {"base_heldout": base, "tables": tables, "selected": selected}
    (OUT / "checkpoint_selection.json").write_text(json.dumps(payload, indent=1))

    md = "# Phase C checkpoint selection（held-out, task book §40）\n\n"
    md += (f"base held-out: reward={base['reward_mean']:+.3f}, invalid={base['invalid_rate']:.3f}, "
           f"FR mismatch={base['fr_mismatch_rate']:.3f}\n\n")
    for tag, rows in tables.items():
        md += f"## {tag}\n\n| step | reward | invalid_rate | FR mismatch | selected |\n|---|---|---|---|---|\n"
        for r in rows:
            sel = "**yes**" if selected[tag] and r["step"] == selected[tag]["step"] else ""
            rew = f"{r['reward_mean']:+.3f}" if r["reward_mean"] is not None else "n/a"
            md += f"| {r['step']} | {rew} | {r['invalid_rate']:.3f} | {r['fr_mismatch']} | {sel} |\n"
        md += "\n"
    (OUT / "checkpoint_selection.md").write_text(md)
    print(f"wrote {OUT / 'checkpoint_selection.md'}")


if __name__ == "__main__":
    main()
