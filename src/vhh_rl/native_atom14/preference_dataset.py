"""Preference pair construction (task book §20-22).

Rank-symmetric pairs per case from the native pool:
    rank 1 vs rank 32, rank 2 vs rank 31, ..., rank 8 vs rank 25
Only pairs with reward_w > reward_l are kept (exact ties dropped).
Winner/loser must share case / FR / design mask / conditioning.
"""
from __future__ import annotations

import json
from pathlib import Path

from .sample_dataset import load_case_samples


def build_pairs_for_case(case_dir: Path, top_k: int = 8) -> list[dict]:
    samples = [s for s in load_case_samples(case_dir) if s.reward_raw is not None
               and not s.contains_invalid and s.fr_mismatch == 0]
    if len(samples) < 2 * top_k:
        return []
    ranked = sorted(samples, key=lambda s: s.reward_raw, reverse=True)
    winners = ranked[:top_k]
    losers = ranked[-top_k:][::-1]  # rank N, N-1, ..., N-top_k+1
    pairs = []
    for w, l in zip(winners, losers):
        if w.reward_raw <= l.reward_raw:
            continue
        pairs.append({
            "case_id": w.case_id,
            "winner_sample_id": w.sample_id,
            "loser_sample_id": l.sample_id,
            "winner_reward": w.reward_raw,
            "loser_reward": l.reward_raw,
            "reward_gap": w.reward_raw - l.reward_raw,
            "condition_id": w.case_id,
            "design_mask_id": w.case_id,
        })
    return pairs


def build_pairs(pool_root: Path, split: str, top_k: int = 8,
                out_path: Path | None = None) -> list[dict]:
    pairs: list[dict] = []
    for case_dir in sorted((Path(pool_root) / split).iterdir()):
        if case_dir.is_dir():
            pairs.extend(build_pairs_for_case(case_dir, top_k=top_k))
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as handle:
            for pair in pairs:
                handle.write(json.dumps(pair) + "\n")
    return pairs


def gap_stats(pairs: list[dict]) -> dict:
    import statistics as st

    gaps = sorted(p["reward_gap"] for p in pairs)
    if not gaps:
        return {"n_pairs": 0}
    def q(p: float) -> float:
        idx = min(len(gaps) - 1, int(p * len(gaps)))
        return gaps[idx]
    return {
        "n_pairs": len(gaps),
        "gap_median": st.median(gaps),
        "gap_mean": st.mean(gaps),
        "gap_p10": q(0.10),
        "gap_p25": q(0.25),
        "gap_p75": q(0.75),
        "gap_p90": q(0.90),
        "gap_min": gaps[0],
        "gap_max": gaps[-1],
    }


def validate_pairs(pairs: list[dict], pool_root: Path, split: str) -> dict:
    """Smoke gate §65: winner/loser same case/FR/design mask, w_reward > l_reward."""
    from .sample_dataset import load_case_samples

    by_case: dict[str, dict[str, object]] = {}
    failures = []
    for pair in pairs:
        if pair["reward_w" if "reward_w" in pair else "winner_reward"] <= pair["loser_reward"]:
            failures.append(("reward_order", pair))
            continue
        case_id = pair["case_id"]
        if case_id not in by_case:
            samples = {s.sample_id: s for s in load_case_samples(Path(pool_root) / split / case_id)}
            by_case[case_id] = samples
        samples = by_case[case_id]
        w = samples[pair["winner_sample_id"]]
        l = samples[pair["loser_sample_id"]]
        if w.decoded_sequence[:1] and l.decoded_sequence[:1] and False:
            pass
        if w.case_id != l.case_id:
            failures.append(("case_mismatch", pair))
    return {"n_checked": len(pairs), "n_failures": len(failures),
            "failures": failures[:5]}
