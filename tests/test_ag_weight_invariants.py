"""Weight invariants for every builder (task book §17)."""
from __future__ import annotations
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.credit.adaptive_granularity import (  # noqa: E402
    adaptive_weights, no_floor_weights, strict_consensus_weights,
    strict_region_weights, validate_weights,
)

BUILDERS = [("no_floor", no_floor_weights, False),
            ("strict_consensus", strict_consensus_weights, False),
            ("strict_region", strict_region_weights, True),
            ("adaptive", adaptive_weights, True)]


def _dataset():
    rows = []
    for i, (d, g) in enumerate([(2.0, 1.0), (1.0, 1.0), (3.0, 2.0),
                                (1.0, -1.0), (-1.0, 1.0), (0.01, 0.01),
                                (0.0, 0.0)]):
        region = "cdr1" if i < 4 else "cdr2"
        rows.append({"position": i, "region": region, "c_drop": d, "c_gain": g,
                     "c_cons": min(d, g) if d > 0 and g > 0 else 0.0})
    for i, (d, g) in enumerate([(1.5, 1.2), (-1.0, 1.0)], start=10):
        rows.append({"position": i, "region": "cdr3", "c_drop": d, "c_gain": g,
                     "c_cons": min(d, g) if d > 0 and g > 0 else 0.0})
    regions = {"cdr1": {"G_drop": 2.0, "G_gain": 1.0},
               "cdr2": {"G_drop": 1.0, "G_gain": -1.0},
               "cdr3": {"G_drop": 0.02, "G_gain": 0.02}}
    return rows, regions


def test_builders_return_valid_or_ineligible():
    rows, regions = _dataset()
    diff = [r["position"] for r in rows]
    for name, builder, needs_regions in BUILDERS:
        out = builder(rows, regions) if needs_regions else builder(rows)
        if out.eligible:
            ok, reason = validate_weights(out.weights, diff)
            assert ok, f"{name}: {reason}"
            assert all(w >= 0 for w in out.weights.values())
            assert sum(out.weights.values()) == pytest.approx(1.0, abs=1e-6)
        else:
            assert out.weights == {}, name


def test_validate_rejects_broken_weights():
    diff = [1, 2, 3]
    assert validate_weights({}, diff)[0] is False
    assert validate_weights({1: 0.5, 9: 0.5}, diff)[0] is False
    assert validate_weights({1: -0.5, 2: 1.5}, diff)[0] is False
    assert validate_weights({1: 0.5}, diff)[0] is False
    assert validate_weights({1: float("nan")}, diff)[0] is False
    assert validate_weights({1: 0.5, 2: 0.5}, diff)[0] is True


def test_weights_never_on_non_differing_positions():
    rows, regions = _dataset()
    diff = {r["position"] for r in rows}
    for name, builder, needs_regions in BUILDERS:
        out = builder(rows, regions) if needs_regions else builder(rows)
        assert set(out.weights) <= diff, name


def test_builders_never_invent_positions():
    """Weights can only live on supplied differing rows (same-residue
    positions are absent from res_rows by construction and must stay absent)."""
    rows, regions = _dataset()
    given = {r["position"] for r in rows}
    for name, builder, needs_regions in BUILDERS:
        out = builder(rows, regions) if needs_regions else builder(rows)
        assert set(out.weights) <= given, name
    for missing in (-1, 99, 999):
        assert missing not in adaptive_weights(rows, regions).weights
