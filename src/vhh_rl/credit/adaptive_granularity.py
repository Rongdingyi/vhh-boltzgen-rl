"""AG-CF-DPO weight construction (AG task book §16-§28, §35-§43, §68).

All methods are pure per-pair weight builders: ``{position -> weight}`` over
the pair's *differing* CDR positions, with NO uniform fallback anywhere.  An
ineligible pair returns ``weights == {}`` and ``eligible == False``; the
training pipeline must drop such pairs instead of re-weighting them.

Design rules taken verbatim from the task book:

* ``TOL = 0.05`` robust signs for residues and regions (§9/§10);
* ``RHO = 0.70`` residue-mode reliability threshold (§37, pre-registered);
* every eligible pair has non-negative weights on differing positions only,
  summing to 1 (§17);
* each changed region is exactly one of ``residue`` / ``region`` / ``abstain``
  (§43) — never a mix inside one region.

This module must not import or modify ``credit.controls`` (historical CF
baseline) nor the trainer.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

TOL = 0.05
RHO = 0.70

# canonical keys
STABLE_POSITIVE = "stable_positive"
STABLE_NEGATIVE = "stable_negative"
SIGN_FLIP = "sign_flip"
WEAK = "weak"

REGION_STABLE_POSITIVE = "stable_positive_region"
REGION_STABLE_NEGATIVE = "stable_negative_region"
REGION_FLIP = "region_flip"
REGION_WEAK = "region_weak"

MODE_RESIDUE = "residue"
MODE_REGION = "region"
MODE_ABSTAIN = "abstain"


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

def robust_sign(x: float | None, tol: float = TOL) -> int:
    """+1 / -1 / 0 with an explicit tolerance band (never sign of tiny floats)."""
    if x is None:
        return 0
    try:
        value = float(x)
    except (TypeError, ValueError):
        return 0
    if math.isnan(value):
        return 0
    if value > tol:
        return 1
    if value < -tol:
        return -1
    return 0


def classify_residue(c_drop: float | None, c_gain: float | None,
                     tol: float = TOL) -> str:
    """Residue robust class from the bidirectional credit (§9)."""
    s_drop, s_gain = robust_sign(c_drop, tol), robust_sign(c_gain, tol)
    if s_drop == 1 and s_gain == 1:
        return STABLE_POSITIVE
    if s_drop == -1 and s_gain == -1:
        return STABLE_NEGATIVE
    if s_drop * s_gain == -1:
        return SIGN_FLIP
    return WEAK


def classify_region(g_drop: float | None, g_gain: float | None,
                    tol: float = TOL) -> str:
    """Region robust class from the group counterfactual (§10)."""
    s_drop, s_gain = robust_sign(g_drop, tol), robust_sign(g_gain, tol)
    if s_drop == 1 and s_gain == 1:
        return REGION_STABLE_POSITIVE
    if s_drop == -1 and s_gain == -1:
        return REGION_STABLE_NEGATIVE
    if s_drop * s_gain == -1:
        return REGION_FLIP
    return REGION_WEAK


# ---------------------------------------------------------------------------
# result type
# ---------------------------------------------------------------------------

@dataclass
class WeightBuildResult:
    weights: dict[int, float] = field(default_factory=dict)
    eligible: bool = False
    modes: dict[str, str] = field(default_factory=dict)
    reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.eligible:
            self.weights = {}
        else:
            self.weights = {int(p): float(w) for p, w in self.weights.items()
                            if float(w) > 0.0}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _num(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(out) else out


def _row_pos(row: Mapping[str, Any]) -> int:
    return int(row["position"])


def _row_region(row: Mapping[str, Any]) -> str:
    return str(row.get("region") or "")


def weight_entropy(weights: Mapping[int, float]) -> float:
    total = sum(float(w) for w in weights.values() if float(w) > 0)
    if total <= 0:
        return 0.0
    return -sum((float(w) / total) * math.log(float(w) / total)
                for w in weights.values() if float(w) > 0)


def _normalize(raw: Mapping[int, float], reason_zero: str,
               eps: float = 1e-12) -> WeightBuildResult:
    """Normalize raw masses; ``eps`` is a numerical epsilon ONLY.

    The classification tolerance ``TOL`` must never be used here: No-floor
    abstains iff the total consistent credit is exactly zero, so tiny-but-
    positive credits still produce weights (task book §19, review item 4).
    """
    values = {int(p): float(w) for p, w in raw.items() if float(w) > 0}
    total = sum(values.values())
    if total <= eps:
        return WeightBuildResult(weights={}, eligible=False, reason=reason_zero)
    weights = {p: v / total for p, v in sorted(values.items())}
    return WeightBuildResult(weights=weights, eligible=True, reason="ok")


def _residue_index(res_rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {_row_pos(r): r for r in res_rows}


def _differing_positions(res_rows: Sequence[Mapping[str, Any]]) -> list[int]:
    return sorted({_row_pos(r) for r in res_rows})


def _positions_by_region(res_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for r in res_rows:
        out.setdefault(_row_region(r), []).append(_row_pos(r))
    return {k: sorted(v) for k, v in out.items()}


def _stable_positive_share(res_rows: Sequence[Mapping[str, Any]],
                           tol: float) -> tuple[dict[str, int], dict[str, int]]:
    n_diff: dict[str, int] = {}
    n_pos: dict[str, int] = {}
    for r in res_rows:
        region = _row_region(r)
        n_diff[region] = n_diff.get(region, 0) + 1
        if classify_residue(r.get("c_drop"), r.get("c_gain"), tol) == STABLE_POSITIVE:
            n_pos[region] = n_pos.get(region, 0) + 1
    return n_diff, n_pos


# ---------------------------------------------------------------------------
# V1: no-floor CF (§19)
# ---------------------------------------------------------------------------

def no_floor_weights(res_rows: Sequence[Mapping[str, Any]],
                     tol: float = TOL) -> WeightBuildResult:
    """s_i = max(c_cons, 0); normalize; ineligible when all zero."""
    raw = {_row_pos(r): max(_num(r.get("c_cons")), 0.0) for r in res_rows}
    out = _normalize(raw, "no_floor:sum_c_cons==0")
    out.diagnostics = {"n_diff": len(raw),
                       "n_active": len(out.weights),
                       "entropy": weight_entropy(out.weights)}
    return out


# ---------------------------------------------------------------------------
# V2: strict-consensus CF (§20)
# ---------------------------------------------------------------------------

def strict_consensus_weights(res_rows: Sequence[Mapping[str, Any]],
                             tol: float = TOL) -> WeightBuildResult:
    """Only robust stable-positive residues; s_i = min(c_drop, c_gain)."""
    raw: dict[int, float] = {}
    n_pos = 0
    for r in res_rows:
        if classify_residue(r.get("c_drop"), r.get("c_gain"), tol) != STABLE_POSITIVE:
            continue
        n_pos += 1
        raw[_row_pos(r)] = min(_num(r.get("c_drop")), _num(r.get("c_gain")))
    out = _normalize(raw, "strict_consensus:no_stable_positive")
    out.diagnostics = {"n_diff": len(res_rows), "n_stable_positive": n_pos,
                       "n_active": len(out.weights),
                       "entropy": weight_entropy(out.weights)}
    return out


# ---------------------------------------------------------------------------
# V3: strict region-only (§21)
# ---------------------------------------------------------------------------

def strict_region_weights(res_rows: Sequence[Mapping[str, Any]],
                          region_rows: Mapping[str, Mapping[str, Any]],
                          tol: float = TOL) -> WeightBuildResult:
    """Uniform per-residue mass inside robust stable-positive regions only."""
    by_region = _positions_by_region(res_rows)
    raw: dict[int, float] = {}
    region_mass: dict[str, float] = {}
    for region, positions in by_region.items():
        row = region_rows.get(region)
        if row is None or not positions:
            continue
        if classify_region(row.get("G_drop"), row.get("G_gain"), tol) != REGION_STABLE_POSITIVE:
            continue
        a_r = max(min(_num(row.get("G_drop")), _num(row.get("G_gain"))), 0.0)
        if a_r <= 0:
            continue
        per = a_r / len(positions)
        region_mass[region] = a_r
        for p in positions:
            raw[p] = raw.get(p, 0.0) + per
    out = _normalize(raw, "strict_region:no_stable_positive_region")
    out.diagnostics = {"n_diff": len(res_rows), "n_regions": len(by_region),
                       "n_active": len(out.weights),
                       "region_mass": region_mass,
                       "entropy": weight_entropy(out.weights)}
    return out


# ---------------------------------------------------------------------------
# Adaptive granularity (§35-§43)
# ---------------------------------------------------------------------------

def adaptive_weights(res_rows: Sequence[Mapping[str, Any]],
                     region_rows: Mapping[str, Mapping[str, Any]],
                     rho: float = RHO, tol: float = TOL) -> WeightBuildResult:
    """Per-region residue / region / abstain decision, then pair normalization.

    Exactly one mode per region (§43): residue keeps only stable-positive
    residue credit; region spreads the group strength uniformly over that
    region's differing residues; abstain contributes nothing.
    """
    by_region = _positions_by_region(res_rows)
    index = _residue_index(res_rows)
    raw: dict[int, float] = {}
    modes: dict[str, str] = {}
    n_residue = n_region = n_abstain = 0
    raw_residue_mass = raw_region_mass = 0.0
    for region in sorted(by_region):
        positions = by_region[region]
        stable = [p for p in positions
                  if classify_residue(index[p].get("c_drop"),
                                      index[p].get("c_gain"), tol) == STABLE_POSITIVE]
        share = (len(stable) / len(positions)) if positions else 0.0
        if share >= rho and stable:
            modes[region] = MODE_RESIDUE
            n_residue += 1
            for p in stable:
                s_i = min(_num(index[p].get("c_drop")), _num(index[p].get("c_gain")))
                raw[p] = raw.get(p, 0.0) + max(s_i, 0.0)
                raw_residue_mass += max(s_i, 0.0)
            continue
        row = region_rows.get(region)
        if row is not None and \
                classify_region(row.get("G_drop"), row.get("G_gain"), tol) == \
                REGION_STABLE_POSITIVE:
            modes[region] = MODE_REGION
            n_region += 1
            a_r = max(min(_num(row.get("G_drop")), _num(row.get("G_gain"))), 0.0)
            if a_r > 0:
                per = a_r / len(positions)
                raw_region_mass += a_r
                for p in positions:
                    raw[p] = raw.get(p, 0.0) + per
            continue
        modes[region] = MODE_ABSTAIN
        n_abstain += 1
    out = _normalize(raw, "adaptive:all_regions_abstain")
    out.modes = modes
    out.diagnostics = {
        "n_diff": len(res_rows),
        "n_regions": len(by_region),
        "n_residue_mode_regions": n_residue,
        "n_region_mode_regions": n_region,
        "n_abstain_regions": n_abstain,
        "n_active_positions": len(out.weights),
        "active_fraction": (len(out.weights) / len(res_rows)) if res_rows else 0.0,
        "weight_entropy": weight_entropy(out.weights),
        "raw_residue_mass": raw_residue_mass,
        "raw_region_mass": raw_region_mass,
    }
    return out


# ---------------------------------------------------------------------------
# controls (§44)
# ---------------------------------------------------------------------------

def shuffle_pair_weights(weights: Mapping[int, float],
                         differing_positions: Iterable[int],
                         rng: random.Random) -> dict[int, float]:
    """Permute the *full* weight multiset across the pair's differing positions.

    The multiset includes the zeros of inactive differing positions, so the
    permutation is a genuine random re-assignment of weight values to
    positions.  This keeps the multiset, entropy, active count and eligibility
    identical while changing the position correspondence — including the
    important case ``active == all differing positions`` where a
    shuffle-of-active-values-only would silently be the identity.
    """
    positions = sorted({int(p) for p in differing_positions})
    outside = {int(p) for p in weights} - set(positions)
    if outside:
        raise ValueError(f"weights on positions outside the differing set: "
                         f"{sorted(outside)[:5]}")
    if not positions:
        return {}
    values = [float(weights.get(p, 0.0)) for p in positions]
    rng.shuffle(values)
    return {p: w for p, w in zip(positions, values) if w > 0}


# ---------------------------------------------------------------------------
# validation (§17)
# ---------------------------------------------------------------------------

def validate_weights(weights: Mapping[int, float],
                     differing_positions: Iterable[int],
                     tol: float = 1e-6) -> tuple[bool, str]:
    """Check the eligible-pair invariant.  ``weights == {}`` is invalid here;
    ineligible pairs are represented by ``WeightBuildResult.eligible == False``.
    """
    diff = {int(p) for p in differing_positions}
    keys = {int(p) for p in weights}
    if not keys:
        return False, "empty weights (ineligible pairs must be dropped, not trained)"
    outside = keys - diff
    if outside:
        return False, f"weights on non-differing positions {sorted(outside)[:5]}"
    total = 0.0
    for p, w in weights.items():
        value = float(w)
        if math.isnan(value) or math.isinf(value):
            return False, f"non-finite weight at {p}"
        if value < -tol:
            return False, f"negative weight at {p}: {value}"
        total += value
    if abs(total - 1.0) > 1e-6:
        return False, f"weights sum to {total:.8f}, expected 1"
    return True, "ok"


# task-book §16 aliases
build_no_floor_weights = no_floor_weights
build_strict_consensus_weights = strict_consensus_weights
build_strict_region_weights = strict_region_weights
build_adaptive_weights = adaptive_weights
validate_pair_weights = validate_weights
