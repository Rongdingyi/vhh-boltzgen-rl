#!/usr/bin/env python
"""Phase B/C pilot training: variants -> run_weighted_dpo (task book §29-§31)."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

import _common as C  # noqa: E402

ARMS = {
    "current": dict(kind="current_cf", variant="cf"),
    "nofloor": dict(kind="ag", variant="no_floor"),
    "strict": dict(kind="ag", variant="strict_consensus"),
    "region": dict(kind="ag", variant="strict_region"),
    "adaptive": dict(kind="ag", variant="adaptive"),
    "shuffle": dict(kind="ag", variant="adaptive_shuffle"),
    "eligible-cf": dict(kind="eligible_cf", variant="cf"),
}
STEPS = 100
SEED = 20260913
GATED_ARMS = {"adaptive", "shuffle", "eligible-cf"}
GATE_A = C.AUDIT_DIR / "audit_summary.json"
GATE_B = C.PILOT_DIR / "gate_b.json"


def resolve_arm_paths(arm: str):
    """(pairs_path, weights_path, trainer variant) for one pilot arm."""
    spec = ARMS[arm]
    if spec["kind"] == "current_cf":
        return (C.WEIGHTS_DIR / "pairs_current_cf_pilot4.jsonl",
                C.WEIGHTS_DIR / "current_cf_pilot4_weights.json", "cf")
    if spec["kind"] == "eligible_cf":
        return (C.WEIGHTS_DIR / "pairs_adaptive_eligible_cf_pilot4.jsonl",
                C.CURRENT_WEIGHTS, "cf")
    variant = spec["variant"]
    return (C.WEIGHTS_DIR / f"pairs_{variant}_pilot4.jsonl",
            C.WEIGHTS_DIR / "ag_weights.json", variant)


def require_gate_a(override: str | None) -> None:
    """No GPU pilot before Gate A passes (task book §14, review 9)."""
    payload = json.loads(GATE_A.read_text()) if GATE_A.is_file() else None
    if payload and payload.get("gate_a_pass") is True:
        return
    if override:
        print(f"[warn] Gate A override: {override}", flush=True)
        return
    raise SystemExit(
        f"Gate A has not passed ({GATE_A}); pilot training blocked. "
        "Run ag-audit first, or pass --override-gate REASON.")


def require_gate_b(override: str | None) -> None:
    """Phase C arms are blocked until Gate B passed (task book §92, review 9)."""
    payload = json.loads(GATE_B.read_text()) if GATE_B.is_file() else None
    if payload and payload.get("pass") is True:
        return
    if override:
        print(f"[warn] Gate B override: {override}", flush=True)
        return
    raise SystemExit(
        f"Gate B has not passed ({GATE_B}); Phase C arms are blocked. "
        "Run Phase B + make_pilot_report first, or pass --override-gate REASON.")


def _materialize_current_cf(pairs: list[dict]) -> tuple[Path, Path]:
    """Filtered current-CF pilot support (same protocol as cf_dpo_mini, §30).

    Inputs live under WEIGHTS_DIR, training output under PILOT_DIR/<arm>:
    the run-dir cleanup must never delete its own inputs (review P0).
    """
    C.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    keep = set(C.PILOT_TRAIN_CASES)
    pilot_pairs = [p for p in pairs if p["case_id"] in keep]
    weights_all = C.load_current_weights()
    weights = {"eta": weights_all.get("eta", 0.75), "seed": weights_all.get("seed"),
               "pairs": {k: v for k, v in weights_all["pairs"].items()
                         if v["case_id"] in keep}}
    pairs_path = C.WEIGHTS_DIR / "pairs_current_cf_pilot4.jsonl"
    weights_path = C.WEIGHTS_DIR / "current_cf_pilot4_weights.json"
    pairs_path.write_text("".join(json.dumps(p) + "\n" for p in pilot_pairs))
    weights_path.write_text(json.dumps(weights, indent=1))
    return pairs_path, weights_path


def arm_output_dir(arm: str) -> Path:
    return C.PILOT_DIR / arm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=list(ARMS))
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--override-gate", default=None,
                        help="reason string; only for deliberate manual overrides")
    args = parser.parse_args()
    spec = ARMS[args.arm]
    pairs = C.load_pairs()

    if spec["kind"] == "current_cf":
        _materialize_current_cf(pairs)
    require_gate_a(args.override_gate)
    if args.arm in GATED_ARMS:
        require_gate_b(args.override_gate)
    pairs_path, weights_path, variant = resolve_arm_paths(args.arm)
    if not pairs_path.is_file():
        raise SystemExit(f"missing {pairs_path}; run ag-build-weights + "
                         f"ag-validate-weights first")
    pilot_pairs = [json.loads(l) for l in pairs_path.open()]
    counts = {case: sum(1 for p in pilot_pairs if p["case_id"] == case)
              for case in C.PILOT_TRAIN_CASES}
    missing = [case for case, n in counts.items() if n == 0]
    if missing:
        raise SystemExit(f"arm={args.arm}: pilot cases with 0 pairs: {missing} "
                         f"(no silent case drop)")

    from vhh_rl.native_atom14.weighted_dpo import run_weighted_dpo

    out_dir = arm_output_dir(args.arm)
    shutil.rmtree(out_dir, ignore_errors=True)   # no mixed old/new logs (review 6)
    summary = run_weighted_dpo(
        base_checkpoint=C.BASE_CKPT,
        pairs_path=pairs_path,
        pool_root=C.POOL,
        conditioning_dir=C.POOL / "conditioning",
        weights_path=weights_path,
        output_dir=out_dir,
        manifest_path=C.MANIFEST,
        variant=variant,
        beta=10.0,
        lr=1e-5,
        max_steps=args.steps,
        checkpoint_every=50,
        seed=args.seed,
        log_tag=f"ag-{args.arm}",
    )
    (out_dir / "pilot_meta.json").write_text(json.dumps(
        {"arm": args.arm, "variant": variant, "pairs_path": str(pairs_path),
         "weights_path": str(weights_path), "steps": args.steps, "seed": args.seed,
         "pilot_case_counts": counts}, indent=1))
    print(json.dumps({"arm": args.arm, "variant": variant,
                      "pilot_case_counts": counts,
                      "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
