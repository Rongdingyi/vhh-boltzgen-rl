"""Gate thresholds and checkers (task book §20-§56).

Pure functions so the gate logic is unit-testable without any model code.
"""
from __future__ import annotations

import hashlib
import json
import statistics as st
from pathlib import Path

GATE1_MIN_VALID_RATE = 0.75
GATE1_MIN_UNIQUE = 3
GATE1_MIN_REWARD_STD = 0.15
GATE1_MIN_BEST_MINUS_MEDIAN = 0.30
GATE1_MIN_TEACHER_HAMMING = 1
GATE1_GROUPS_PER_PROGRESS = 8
GATE1_MIN_GROUPS_PASS = 6

GATE2_MIN_REPLAY_ABS = 1e-5
GATE2_MSE_RATIO = 0.50
GATE2_MEDIAN_MSE_RATIO = 0.40
GATE2_MIN_IMPROVED = 10
GATE2_MIN_DOWNSTREAM = 6
GATE2_SAFETY_SLACK = 0.20
GATE2_MIN_SAFE = 8

GATE3_MIN_D_A = 0.50
GATE3_MIN_CASES_D_GE_A = 5
GATE3_MIN_D_B = 0.25
GATE3_MIN_D_C = 0.25
GATE3_MAX_INVALID = 0.01


def protocol_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def guard_protocol(artifact: dict, frozen_path: str | Path) -> None:
    """Every gate artifact must carry the frozen protocol hash (§61)."""
    expected = protocol_hash(frozen_path)
    actual = artifact.get("protocol_sha256")
    if actual != expected:
        raise RuntimeError(
            f"protocol hash mismatch: artifact={actual} frozen={expected}; "
            "re-freeze and re-run instead of editing the protocol")


# ---------------------------------------------------------------- Gate 1 ---

def gate1_group_pass(metrics: dict) -> bool:
    return (metrics["valid_rate"] >= GATE1_MIN_VALID_RATE
            and metrics["n_unique_valid_sequences"] >= GATE1_MIN_UNIQUE
            and metrics["reward_std"] >= GATE1_MIN_REWARD_STD
            and (metrics["best_minus_median"] or 0.0) >= GATE1_MIN_BEST_MINUS_MEDIAN
            and (metrics["teacher_vs_median_hamming"] or 0) >= GATE1_MIN_TEACHER_HAMMING)


def gate1_progress_pass(group_metrics: list[dict]) -> dict:
    n_pass = sum(1 for m in group_metrics if gate1_group_pass(m))
    median_valid = st.median(m["valid_rate"] for m in group_metrics) if group_metrics else 0.0
    median_unique = st.median(m["n_unique_valid_sequences"] for m in group_metrics) if group_metrics else 0.0
    bmm = [m["best_minus_median"] for m in group_metrics if m["best_minus_median"] is not None]
    median_bmm = st.median(bmm) if bmm else 0.0
    ok = (n_pass >= GATE1_MIN_GROUPS_PASS
          and median_valid >= GATE1_MIN_VALID_RATE
          and median_unique >= GATE1_MIN_UNIQUE
          and median_bmm >= GATE1_MIN_BEST_MINUS_MEDIAN)
    return {"pass": ok, "n_groups": len(group_metrics), "n_groups_pass": n_pass,
            "median_valid_rate": median_valid, "median_unique": median_unique,
            "median_best_minus_median": median_bmm}


# ---------------------------------------------------------------- Gate 2 ---

def gate2_replay_pass(maxdiffs: list[float]) -> bool:
    return len(maxdiffs) > 0 and all(d < GATE2_MIN_REPLAY_ABS for d in maxdiffs)


def gate2_target_pass(rows: list[dict]) -> bool:
    return len(rows) > 0 and all(
        (not r["contains_invalid"]) and r["fr_mismatch"] == 0
        and r.get("all_changed_match") for r in rows)


def gate2_learnability_pass(ratios: list[float]) -> bool:
    if not ratios:
        return False
    improved = sum(1 for r in ratios if r <= GATE2_MSE_RATIO)
    return improved >= GATE2_MIN_IMPROVED and st.median(ratios) <= GATE2_MEDIAN_MSE_RATIO


def gate2_downstream_pass(hamming_before: list[int], hamming_after: list[int]) -> bool:
    if not hamming_before:
        return False
    improved = sum(1 for b, a in zip(hamming_before, hamming_after) if a < b)
    return improved >= GATE2_MIN_DOWNSTREAM


def gate2_safety_pass(postfit: list[float], peer: list[float]) -> bool:
    if not postfit:
        return False
    deltas = [p - q for p, q in zip(postfit, peer)]
    safe = sum(1 for d in deltas if d >= -GATE2_SAFETY_SLACK)
    return st.median(deltas) >= 0 and safe >= GATE2_MIN_SAFE


def gate2_verdict(rows: list[dict]) -> dict:
    """rows carry replay_maxdiff, carrier_*, mse_ratio, hamming_before/after,
    postfit_reward, peer_reward (one row per record)."""
    replay = gate2_replay_pass([r["replay_maxdiff"] for r in rows])
    target = gate2_target_pass([{"contains_invalid": r["carrier_invalid"],
                                 "fr_mismatch": r["carrier_fr"],
                                 "all_changed_match": r["carrier_all_match"]}
                                for r in rows])
    learn = gate2_learnability_pass([r["mse_ratio"] for r in rows])
    down = gate2_downstream_pass([r["hamming_before"] for r in rows],
                                 [r["hamming_after"] for r in rows])
    safety = gate2_safety_pass([r["postfit_reward"] for r in rows],
                               [r["peer_reward"] for r in rows])
    return {"A_replay": replay, "B_target": target, "C_learnability": learn,
            "D_downstream": down, "E_safety": safety,
            "pass": all((replay, target, learn, down, safety))}


# ---------------------------------------------------------------- Gate 3 ---

def gate3_verdict(arm_rewards: dict[str, float], per_case: dict[str, dict],
                  invalid_rate: float, fr_mismatch: int) -> dict:
    """arm_rewards: {"A":…, "B":…, "C":…, "D":…}; per_case[case][arm]."""
    d, a, b, c = (arm_rewards.get(k) for k in ("D", "A", "B", "C"))
    if None in (d, a, b, c):
        return {"verdict": "NOT_RUN"}
    delta_a, delta_b, delta_c = d - a, d - b, d - c
    cases = [c_id for c_id, v in per_case.items()
             if v.get("D") is not None and v.get("A") is not None]
    n_d_ge_a = sum(1 for c_id in cases if per_case[c_id]["D"] >= per_case[c_id]["A"])
    strong = (delta_a >= GATE3_MIN_D_A
              and n_d_ge_a >= GATE3_MIN_CASES_D_GE_A
              and delta_b >= GATE3_MIN_D_B
              and delta_c >= GATE3_MIN_D_C
              and invalid_rate <= GATE3_MAX_INVALID
              and fr_mismatch == 0)
    if d <= a:
        verdict = "NO_GO"
    elif strong:
        verdict = "STRONG_GO"
    else:
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, "delta_D_A": delta_a, "delta_D_B": delta_b,
            "delta_D_C": delta_c, "cases_D_ge_A": n_d_ge_a, "n_cases": len(cases),
            "invalid_rate": invalid_rate, "fr_mismatch": fr_mismatch}
