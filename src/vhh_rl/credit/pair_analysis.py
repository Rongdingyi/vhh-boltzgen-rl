"""Phase A0 statistics: pair Hamming distances and reward-gap correlations."""
from __future__ import annotations

import statistics as st
from typing import Sequence


def hamming_positions(a: str, b: str, positions: Sequence[int]) -> list[int]:
    if len(a) != len(b):
        raise ValueError("sequences differ in length")
    return [int(i) for i in positions if a[i] != b[i]]


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3:
        return None
    return _pearson(_rank(xs), _rank(ys))


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(len(sorted_values) - 1, int(round(p * (len(sorted_values) - 1))))
    return sorted_values[idx]


def hamming_stats(pairs: Sequence[dict], sequences: dict[str, str],
                  design_positions: dict[str, tuple[int, ...]]) -> dict:
    """pairs: rows with winner_sample_id/loser_sample_id/reward_gap/case_id."""
    diffs: list[int] = []
    gaps: list[float] = []
    total_diff = 0
    for pair in pairs:
        case = pair["case_id"]
        w = sequences[pair["winner_sample_id"]]
        l = sequences[pair["loser_sample_id"]]
        pos = hamming_positions(w, l, design_positions[case])
        diffs.append(len(pos))
        gaps.append(float(pair["reward_gap"]))
        total_diff += len(pos)
    srt = sorted(diffs)
    return {
        "n_pairs": len(pairs),
        "total_differing_positions": total_diff,
        "hamming_mean": st.mean(diffs),
        "hamming_median": st.median(diffs),
        "hamming_min": min(diffs),
        "hamming_max": max(diffs),
        "hamming_p10": _percentile(srt, 0.10),
        "hamming_p25": _percentile(srt, 0.25),
        "hamming_p75": _percentile(srt, 0.75),
        "hamming_p90": _percentile(srt, 0.90),
        "reward_gap_mean": st.mean(gaps),
        "reward_gap_median": st.median(gaps),
        "pearson_hamming_gap": _pearson(diffs, gaps),
        "spearman_hamming_gap": _spearman(diffs, gaps),
    }
