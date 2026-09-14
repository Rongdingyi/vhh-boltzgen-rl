#!/usr/bin/env python
"""Select the best second-reward checkpoint per arm on held-out (ESM-C cdr_pll),
then materialize <arm>_heldout_scored.jsonl for the final comparison.

Held-out rule: highest mean cdr_pll with invalid==0 (and FR mismatch==0 when
the field is present).
"""
from __future__ import annotations

import glob
import json
import shutil
import statistics as st
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
SEC = ROOT / "runs/paper_stage/second_reward"
SWEEP = SEC / "sweep"
ARMS = ["uniform", "cf"]


def main() -> None:
    selection = {"arms": {}}
    for arm in ARMS:
        rows = []
        for f in sorted(glob.glob(str(SWEEP / f"srsweep_{arm}_s*_scored.jsonl"))):
            step = int(Path(f).stem.split("_s")[1][:4])
            data = [json.loads(l) for l in open(f)]
            ok = [r for r in data if r.get("cdr_pll") is not None
                  and not r.get("contains_invalid")
                  and int(r.get("fr_mismatch", 0)) == 0]
            rows.append({"step": step, "n_ok": len(ok), "n_total": len(data),
                         "mean_cdr_pll": st.mean([r["cdr_pll"] for r in ok]) if ok else None,
                         "path": f})
        rows.sort(key=lambda r: r["step"])
        ok = [r for r in rows if r["mean_cdr_pll"] is not None]
        best = max(ok, key=lambda r: r["mean_cdr_pll"]) if ok else None
        selection["arms"][arm] = {"rows": [{k: v for k, v in r.items() if k != "path"}
                                           for r in rows],
                                  "selected_step": best["step"] if best else None,
                                  "selected_mean_cdr_pll": best["mean_cdr_pll"] if best else None}
        print(f"[{arm}] " + " ".join(f"s{r['step']}:{r['mean_cdr_pll']:+.4f}"
                                     if r["mean_cdr_pll"] is not None else f"s{r['step']}:n/a"
                                     for r in rows) + f" -> {best['step'] if best else None}")
        if best:
            shutil.copyfile(best["path"], SEC / f"{arm}_heldout_scored.jsonl")
    (SEC / "checkpoint_selection.json").write_text(json.dumps(selection, indent=1))
    print(f"wrote {SEC / 'checkpoint_selection.json'}")


if __name__ == "__main__":
    main()
