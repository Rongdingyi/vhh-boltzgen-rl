"""V3 Strict Region-only (task book §21/§22/§28)."""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.credit.adaptive_granularity import (  # noqa: E402
    strict_region_weights, validate_weights,
)


def _rows():
    rows = []
    for pos in (1, 2):                      # CDR1: 2 differing
        rows.append({"position": pos, "region": "cdr1", "c_drop": 1.0,
                     "c_gain": 1.0, "c_cons": 1.0})
    for pos in (5, 6, 7, 8):                # CDR3: 4 differing
        rows.append({"position": pos, "region": "cdr3", "c_drop": 1.0,
                     "c_gain": 1.0, "c_cons": 1.0})
    return rows


def _regions():
    return {"cdr1": {"G_drop": 3.0, "G_gain": 2.5, "G_cons": 2.0},
            "cdr3": {"G_drop": 1.2, "G_gain": 1.0, "G_cons": 1.0}}


def test_region_mass_is_uniform_inside_region():
    out = strict_region_weights(_rows(), _regions())
    assert out.eligible
    # A_r = min(G_drop, G_gain): cdr1 = 2.5 over 2, cdr3 = 1.0 over 4
    raw = {1: 2.5 / 2, 2: 2.5 / 2, 5: 1.0 / 4, 6: 1.0 / 4,
           7: 1.0 / 4, 8: 1.0 / 4}
    total = sum(raw.values())
    for pos, value in raw.items():
        assert out.weights[pos] == pytest.approx(value / total)
    ok, reason = validate_weights(out.weights, [1, 2, 5, 6, 7, 8])
    assert ok, reason


def test_unstable_region_excluded_entirely():
    regions = {"cdr1": {"G_drop": 3.0, "G_gain": 2.5},
               "cdr3": {"G_drop": 1.0, "G_gain": -1.0}}
    out = strict_region_weights(_rows(), regions)
    assert set(out.weights) == {1, 2}


def test_no_stable_region_is_ineligible_no_uniform():
    regions = {"cdr1": {"G_drop": 1.0, "G_gain": -1.0},
               "cdr3": {"G_drop": 0.01, "G_gain": 0.01}}
    out = strict_region_weights(_rows(), regions)
    assert out.eligible is False
    assert out.weights == {}
