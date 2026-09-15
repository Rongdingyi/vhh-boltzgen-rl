#!/usr/bin/env python
"""Phase B/C pilot evaluation: historical-4 + all-heldout-8 (task book §32)."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

import _common as C  # noqa: E402

ARMS = ["current", "nofloor", "strict", "region", "adaptive", "shuffle", "eligible-cf"]
CKPT_STEP = 100
NUM_SAMPLES = 8
SEED_OFFSET = 800000


def _evaluate(tag: str, ckpt: Path | None, cases: list[str], run_tag: str) -> dict:
    from vhh_rl.cf_opsd.evaluator import evaluate

    run_root = C.PILOT_DIR / "eval" / run_tag
    shutil.rmtree(run_root, ignore_errors=True)   # stale-dir guard (review 7)
    return evaluate(sorted(cases), ckpt, tag, num_samples=NUM_SAMPLES,
                    seed_offset=SEED_OFFSET, run_root=run_root)


def _summarize(summary: dict, refs: dict[str, float | None]) -> dict:
    cases = summary["cases"]
    n = sum(v["n"] for v in cases.values())
    hist4 = [v["reward_mean"] for c, v in cases.items() if c in C.HELDOUT4
             and v["reward_mean"] is not None]
    payload = {
        "reward_mean": summary["reward_mean"],
        "reward_mean_hist4": (sum(hist4) / len(hist4)) if hist4 else None,
        "per_case": {c: v["reward_mean"] for c, v in cases.items()},
        "invalid": sum(v["n_invalid"] for v in cases.values()),
        "fr": sum(v["n_fr_mismatch"] for v in cases.values()),
        "unique": sum(v["n_unique"] for v in cases.values()),
        "n": n,
    }
    for name in ("base", "current"):
        ref = refs.get(name)                       # scalar reference only
        if ref is None:
            continue
        payload[f"delta_vs_{name}"] = summary["reward_mean"] - ref
        ref_cases = refs.get(f"per_case_{name}") or {}
        wins = ties = losses = 0
        for c, v in cases.items():
            other = ref_cases.get(c)
            if other is None or v["reward_mean"] is None:
                continue
            d = v["reward_mean"] - other
            wins += int(d > 1e-9)
            losses += int(d < -1e-9)
            ties += int(abs(d) <= 1e-9)
        payload[f"wins_vs_{name}"] = f"{wins}/{ties}/{losses}"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="*", default=ARMS)
    parser.add_argument("--with-base", action="store_true", default=True)
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()
    heldout8 = C.heldout8()
    results: dict[str, dict] = {}
    base = _evaluate("ag_base", None, heldout8, args.run_tag or "base")
    results["base"] = _summarize(base, {})
    ref_per_case = {c: v["reward_mean"] for c, v in base["cases"].items()}
    ordered = ["current"] + [a for a in args.arms if a != "current"]
    for arm in ordered:
        ckpt = C.PILOT_DIR / arm / f"checkpoint_{CKPT_STEP:04d}.pt"
        if not ckpt.is_file():
            print(f"[skip] {arm}: missing {ckpt}")
            continue
        summary = _evaluate(f"ag_{arm}", ckpt, heldout8,
                            args.run_tag or arm)
        refs = {"base": results["base"]["reward_mean"],
                "per_case_base": ref_per_case}
        if "current" in results:
            refs["current"] = results["current"]["reward_mean"]
            refs["per_case_current"] = results["current"]["per_case"]
        results[arm] = _summarize(summary, refs)
        if arm == "current" and "shuffle" not in results:
            pass
    out = C.PILOT_DIR / "eval" / "pilot_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged = json.loads(out.read_text()) if out.is_file() else {}
    merged.update(results)
    out.write_text(json.dumps(merged, indent=1))
    print(json.dumps({k: v.get("reward_mean") for k, v in merged.items()}, indent=1))


if __name__ == "__main__":
    main()
