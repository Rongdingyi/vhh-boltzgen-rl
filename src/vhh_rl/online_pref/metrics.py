"""Metrics and query accounting for online preference DPO (task book §55/§56)."""
from __future__ import annotations

import statistics as st

from .pair_builder import changed_design_positions


def reward_query_count(groups) -> int:
    """Number of scored sibling endpoints (= reward scorer sequence queries)."""
    return sum(1 for g in groups for s in g.siblings if s.reward is not None)


def generated_endpoint_count(groups) -> int:
    return sum(len(g.siblings) for g in groups)


def group_stats(groups) -> dict:
    valid = [s for g in groups for s in g.siblings
             if not s.contains_invalid and s.fr_mismatch == 0 and s.reward is not None]
    rewards = [s.reward for s in valid]
    unique = {s.endpoint_sequence for s in valid}
    return {
        "n_groups": len(groups),
        "n_valid_siblings": len(valid),
        "n_unique_sequences": len(unique),
        "reward_std": st.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
    }


def pair_stats(pairs, design_positions_by_case) -> dict:
    if not pairs:
        return {"n_pairs": 0, "mean_pair_hamming": None, "mean_reward_gap": None}
    hamming = [len(changed_design_positions(p.winner_sequence, p.loser_sequence,
                                            design_positions_by_case[p.case_id]))
               for p in pairs]
    gaps = [p.reward_gap for p in pairs]
    return {
        "n_pairs": len(pairs),
        "mean_pair_hamming": st.mean(hamming),
        "median_pair_hamming": st.median(hamming),
        "mean_reward_gap": st.mean(gaps),
        "min_reward_gap": min(gaps),
    }


def case_level_mean(values: list[float]) -> float | None:
    return st.mean(values) if values else None
