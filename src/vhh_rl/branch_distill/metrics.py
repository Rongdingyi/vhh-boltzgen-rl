"""Branch-group metrics (task book §25)."""
from __future__ import annotations

import statistics as st


def design_hamming(seq_a: str, seq_b: str, design_positions) -> int:
    """Number of differing residues over design positions only (§19)."""
    return sum(1 for p in design_positions
               if seq_a[p] != seq_b[p])


def mean_pairwise_hamming(sequences: list[str], design_positions) -> float | None:
    if len(sequences) < 2:
        return None
    total, count = 0, 0
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            total += design_hamming(sequences[i], sequences[j], design_positions)
            count += 1
    return total / count if count else None


def group_metrics(siblings, design_positions) -> dict:
    """Per-group metrics over one BranchGroup's siblings (§25)."""
    valid = [s for s in siblings
             if not s.contains_invalid and s.fr_mismatch == 0 and s.reward is not None]
    n_total = len(siblings)
    rewards = sorted(s.reward for s in valid)
    sequences = [s.endpoint_sequence for s in valid]
    unique = len(set(sequences))
    reward_std = st.pstdev(rewards) if len(rewards) > 1 else 0.0
    reward_mean = st.mean(rewards) if rewards else None
    reward_median = st.median(rewards) if rewards else None
    best = rewards[-1] if rewards else None
    worst = rewards[0] if rewards else None
    best_idx = max(range(len(valid)), key=lambda i: valid[i].reward) if valid else None
    median_idx = _median_index(rewards, valid) if valid else None
    worst_idx = min(range(len(valid)), key=lambda i: valid[i].reward) if valid else None
    teacher_vs_median = (design_hamming(valid[best_idx].endpoint_sequence,
                                        valid[median_idx].endpoint_sequence,
                                        design_positions)
                         if best_idx is not None and median_idx is not None else None)
    teacher_vs_worst = (design_hamming(valid[best_idx].endpoint_sequence,
                                       valid[worst_idx].endpoint_sequence,
                                       design_positions)
                        if best_idx is not None and worst_idx is not None else None)
    return {
        "n_total": n_total,
        "n_valid": len(valid),
        "valid_rate": len(valid) / n_total if n_total else 0.0,
        "n_unique_valid_sequences": unique,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "reward_max": best,
        "reward_median": reward_median,
        "reward_min": worst,
        "best_minus_median": (best - reward_median
                              if best is not None and reward_median is not None else None),
        "best_minus_worst": (best - worst
                             if best is not None and worst is not None else None),
        "mean_pairwise_hamming_design": mean_pairwise_hamming(sequences, design_positions),
        "teacher_vs_median_hamming": teacher_vs_median,
        "teacher_vs_worst_hamming": teacher_vs_worst,
        "best_index": best_idx,
        "median_index": median_idx,
        "worst_index": worst_idx,
    }


def _median_index(sorted_rewards: list[float], valid) -> int:
    target = st.median(sorted_rewards)
    return min(range(len(valid)), key=lambda i: abs(valid[i].reward - target))
