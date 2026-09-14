"""Preference weight construction and controls (task book §32-38).

All weights are per-pair dicts {position -> w} with sum(w) == 1 over the
pair's differing CDR residues (positions where winner and loser differ; §34:
same-residue positions never receive preference weight).
"""
from __future__ import annotations

import random
from typing import Mapping

ETA = 0.75


def cf_weights(credits: Mapping[int, float], n_diff: int,
               eta: float = ETA) -> tuple[dict[int, float], bool]:
    """Consistent-credit weights with uniform floor.

    Returns (weights, fallback_used).  Fallback = uniform over the differing
    residues when all c_cons are zero (§33).
    """
    if n_diff <= 0:
        return {}, True
    floor = (1.0 - eta) / n_diff
    total = sum(c for c in credits.values() if c > 0)
    if total <= 0:
        return {p: 1.0 / n_diff for p in credits}, True
    w = {p: floor + eta * (max(c, 0.0) / total) for p, c in credits.items()}
    s = sum(w.values())
    return {p: v / s for p, v in w.items()}, False


def random_sparse_weights(credits: Mapping[int, float], n_diff: int,
                          rng: random.Random,
                          uniform_floor: bool = False) -> dict[int, float]:
    """B1 control: random residues carry the weight (matched nonzero count).

    k = number of residues with positive consistent credit; k positions are
    drawn uniformly at random; each gets 1/k.  Documented entropy mismatch is
    small because the eta floor already flattens CF weights.
    """
    if n_diff <= 0:
        return {}
    positive = [p for p, c in credits.items() if c > 0]
    k = max(1, len(positive))
    if k >= n_diff:
        return {p: 1.0 / n_diff for p in credits}
    chosen = rng.sample(sorted(credits), k)
    return {p: 1.0 / k for p in chosen}


def shuffle_weights(weights: Mapping[int, float], rng: random.Random) -> dict[int, float]:
    """B2 control: permute the true weight values across the pair's positions."""
    positions = sorted(weights)
    values = [weights[p] for p in positions]
    rng.shuffle(values)
    return dict(zip(positions, values))


def region_weights(region_credits: Mapping[str, float], positions_by_region: Mapping[str, list[int]],
                   floor: float | None = None) -> dict[int, float]:
    """B4 (optional): region-level weights, uniform inside each CDR region."""
    u = {r: max(c, 0.0) for r, c in region_credits.items()}
    total = sum(u.values())
    if total <= 0:
        u = {r: 1.0 for r in region_credits}
        total = sum(u.values())
    out: dict[int, float] = {}
    for region, value in u.items():
        pos = positions_by_region.get(region, [])
        if not pos:
            continue
        per = (value / total) / len(pos)
        for p in pos:
            out[p] = per
    s = sum(out.values()) or 1.0
    return {p: v / s for p, v in out.items()}


def weight_entropy(weights: Mapping[int, float]) -> float:
    import math
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    return -sum((v / total) * math.log(v / total) for v in weights.values() if v > 0)
