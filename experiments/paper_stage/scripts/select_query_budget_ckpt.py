#!/usr/bin/env python
"""Select query-budget-arm checkpoints on the 8 held-out cases (§54 rule)."""
from __future__ import annotations

import glob
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
POOL = ROOT / "runs/native_pool"
OUT = ROOT / "runs/paper_stage/query_budget"
RUNS = ["qb_025", "qb_050", "qb_075"]


def summarize(path: Path) -> dict:
    s = json.loads(path.read_text())
    cases = s["cases"]
    n = sum(v["n_samples"] for v in cases.values())
    return {
        "reward_mean": st.mean([v["reward_mean"] for v in cases.values()
                                if v["reward_mean"] is not None]),
        "invalid_rate": sum(v["n_invalid"] for v in cases.values()) / max(n, 1),
        "fr_mismatch": sum(v["n_fr_mismatch"] for v in cases.values()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base_cases = {}
    for d in sorted((POOL / "heldout").iterdir()):
        if not d.is_dir():
            continue
        rs = [json.loads(l)["reward_raw"] for l in (d / "metadata.jsonl").open()]
        rs = [r for r in rs if r is not None]
        if rs:
            base_cases[d.name] = st.mean(rs)
    result = {"base_heldout_reward": st.mean(base_cases.values()), "runs": {}}
    for run in RUNS:
        rows = []
        for path in sorted(glob.glob(str(POOL / f"eval_summary_paper_{run}_s*_heldout_shard0.json"))):
            step = int(re.search(r"_s(\d+)_heldout", Path(path).name).group(1))
            rows.append({"step": step, **summarize(Path(path))})
        rows.sort(key=lambda r: r["step"])
        ok = [r for r in rows if r["invalid_rate"] <= 0.01 and r["fr_mismatch"] == 0]
        best = max(ok, key=lambda r: r["reward_mean"]) if ok else None
        result["runs"][run] = {"rows": rows,
                               "selected": best["step"] if best else None,
                               "selected_reward": best["reward_mean"] if best else None,
                               "ckpt": f"runs/paper_stage/{run}/checkpoint_{best['step']:04d}.pt"
                                       if best else None}
        print(f"[{run}] " + " ".join(f"s{r['step']}:{r['reward_mean']:+.2f}" for r in rows)
              + f" -> {best['step'] if best else None}")
    (OUT / "checkpoint_selection.json").write_text(json.dumps(result, indent=1))
    print(f"wrote {OUT / 'checkpoint_selection.json'}")


if __name__ == "__main__":
    main()
