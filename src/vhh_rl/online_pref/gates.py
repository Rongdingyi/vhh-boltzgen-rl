"""Gate thresholds and protocol helpers (task book §14/§26/§31/§35/§42/§45)."""
from __future__ import annotations

import hashlib
import json
import statistics as st
from pathlib import Path

PHASE0_MIN_MEAN_DELTA = 0.30
PHASE0_MIN_SEEDS_POSITIVE = 2
PHASE0_MIN_CASES_GE = 4
PHASE1_MIN_SUFFIX_FULL = 0.25
PHASE1_MIN_SUFFIX_PREFIX = 0.50
PHASE1_MAX_PREFIX_FULL = 0.10
PHASE2_MIN_SUFFIX_FULL = 0.30
PHASE3_MIN_P80_P60 = 0.20
PHASE3_MIN_ELIGIBLE_RATIO = 0.75
PHASE4_MIN_MINEDIT_RANK = 0.20
FINAL_MIN_FINAL_OFFLINE = 0.50
FINAL_MIN_FINAL_ONLINE = 0.25
MAX_INVALID = 0.01


def protocol_hash(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def guard_protocol(artifact: dict, frozen_path) -> None:
    expected = protocol_hash(frozen_path)
    if artifact.get("protocol_sha256") != expected:
        raise RuntimeError("protocol hash mismatch: re-freeze instead of editing")


def write_gate(path, *, pass_, protocol_sha256: str, train_seeds, metrics: dict,
               decision: str) -> dict:
    payload = {"pass": bool(pass_), "protocol_sha256": protocol_sha256,
               "train_seeds": list(train_seeds), "metrics": metrics,
               "decision": decision}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1))
    return payload


def phase0_verdict(per_seed_delta: dict[int, float],
                   per_seed_cases_ge: dict[int, int],
                   invalid_rate: float, fr_mismatch: int) -> dict:
    deltas = [per_seed_delta[s] for s in sorted(per_seed_delta)]
    positive = sum(1 for d in deltas if d > 0)
    mean_delta = st.mean(deltas) if deltas else None
    cases_ge = [per_seed_cases_ge[s] for s in sorted(per_seed_cases_ge)]
    median_cases = st.median(cases_ge) if cases_ge else 0
    passed = (mean_delta is not None and mean_delta >= PHASE0_MIN_MEAN_DELTA
              and positive >= PHASE0_MIN_SEEDS_POSITIVE
              and median_cases >= PHASE0_MIN_CASES_GE
              and invalid_rate <= MAX_INVALID and fr_mismatch == 0)
    return {"pass": passed, "mean_delta": mean_delta, "seeds_positive": positive,
            "median_cases_ge": median_cases, "invalid_rate": invalid_rate,
            "fr_mismatch": fr_mismatch,
            "thresholds": {"min_mean_delta": PHASE0_MIN_MEAN_DELTA,
                           "min_seeds_positive": PHASE0_MIN_SEEDS_POSITIVE,
                           "min_cases_ge": PHASE0_MIN_CASES_GE,
                           "max_invalid": MAX_INVALID}}


def signed_seed_mean_std(values: list[float]) -> tuple[float, float]:
    """Seed-level mean ± sample std (task book §55: use stdev, not pstdev)."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return st.mean(values), st.stdev(values)
