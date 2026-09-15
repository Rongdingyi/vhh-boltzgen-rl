"""Adaptive granularity: residue / region / abstain + pair normalization.

Task book §37-§43, §73-§76.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.credit.adaptive_granularity import (  # noqa: E402
    MODE_ABSTAIN, MODE_REGION, MODE_RESIDUE, adaptive_weights, validate_weights,
)


def _res(entries):
    """entries: (position, region, c_drop, c_gain)"""
    return [{"position": p, "region": r, "c_drop": d, "c_gain": g,
             "c_cons": min(d, g) if d > 0 and g > 0 else 0.0}
            for p, r, d, g in entries]


def test_residue_mode_zeroes_flip_weight():
    # §73: 4 differing, 3 stable positive, 1 flip -> rho = 0.75 >= 0.70
    rows = _res([(1, "cdr1", 2.0, 1.0), (2, "cdr1", 1.0, 3.0),
                 (3, "cdr1", 2.0, 2.0), (4, "cdr1", 1.0, -1.0)])
    out = adaptive_weights(rows, {"cdr1": {"G_drop": 0.1, "G_gain": 0.1}})
    assert out.eligible
    assert out.modes == {"cdr1": MODE_RESIDUE}
    assert 4 not in out.weights                      # flip position is zero
    assert set(out.weights) == {1, 2, 3}
    raw = {1: 1.0, 2: 1.0, 3: 2.0}
    total = sum(raw.values())
    for p, v in raw.items():
        assert out.weights[p] == pytest.approx(v / total)
    assert out.diagnostics["raw_residue_mass"] == pytest.approx(4.0)
    assert out.diagnostics["raw_region_mass"] == 0.0


def test_region_mode_spreads_group_mass_uniformly():
    # §74: rho = 0.25, but G_drop/G_gain stable -> region mode
    rows = _res([(1, "cdr3", 1.0, 1.0), (2, "cdr3", 1.0, -1.0),
                 (3, "cdr3", -1.0, 1.0), (4, "cdr3", 0.01, 0.01)])
    out = adaptive_weights(rows, {"cdr3": {"G_drop": 2.0, "G_gain": 1.5}})
    assert out.modes == {"cdr3": MODE_REGION}
    assert set(out.weights) == {1, 2, 3, 4}
    for pos in (1, 2, 3, 4):
        assert out.weights[pos] == pytest.approx(0.25)
    assert out.diagnostics["raw_region_mass"] == pytest.approx(1.5)


def test_abstain_mode_contributes_nothing():
    # §75: region flip -> abstain even though one residue looks stable
    rows = _res([(1, "cdr3", 1.0, 1.0), (2, "cdr3", 1.0, -1.0),
                 (3, "cdr3", -1.0, 1.0), (4, "cdr3", 0.01, 0.01)])
    out = adaptive_weights(rows, {"cdr3": {"G_drop": 2.0, "G_gain": -1.0}})
    assert out.modes == {"cdr3": MODE_ABSTAIN}
    assert out.eligible is False
    assert out.weights == {}


def test_mixed_modes_pair_normalization():
    # §76: residue region (cdr1) + region mode (cdr2) + abstain (cdr3)
    rows = _res([(1, "cdr1", 2.0, 1.0), (2, "cdr1", 1.0, 1.0),
                 (8, "cdr1", 1.0, 1.0), (3, "cdr1", 1.0, -1.0),
                 (4, "cdr2", 1.0, -1.0), (5, "cdr2", -1.0, 1.0),
                 (6, "cdr3", 1.0, -1.0), (7, "cdr3", -1.0, 1.0)])
    regions = {"cdr1": {"G_drop": 0.1, "G_gain": 0.1},
               "cdr2": {"G_drop": 1.0, "G_gain": 1.0},
               "cdr3": {"G_drop": 1.0, "G_gain": -1.0}}
    out = adaptive_weights(rows, regions)
    assert out.modes == {"cdr1": MODE_RESIDUE, "cdr2": MODE_REGION,
                         "cdr3": MODE_ABSTAIN}
    ok, reason = validate_weights(out.weights, [1, 2, 3, 4, 5, 6, 7, 8])
    assert ok, reason
    assert set(out.weights) == {1, 2, 8, 4, 5}          # cdr3 abstains
    # cdr1 raw mass 1+1+1=3 (3/4 = 0.75 >= 0.70 -> residue mode) ;
    # cdr2 raw mass 1.0 over 2 positions (region mode) ; total raw 4
    assert out.weights[1] == pytest.approx(1.0 / 4.0)
    assert out.weights[8] == pytest.approx(1.0 / 4.0)
    assert out.weights[4] == pytest.approx(0.5 / 4.0)


def test_region_exclusivity_no_partial_mixing():
    # §43: when a region is region-mode its stable residues do NOT get
    # residue-mode mass on top; all positions share the same value.
    rows = _res([(1, "cdr1", 5.0, 5.0), (2, "cdr1", 1.0, -1.0),
                 (3, "cdr1", -1.0, 1.0), (4, "cdr1", -1.0, -1.0)])
    out = adaptive_weights(rows, {"cdr1": {"G_drop": 2.0, "G_gain": 2.0}})
    assert out.modes == {"cdr1": MODE_REGION}
    assert out.weights[1] == pytest.approx(out.weights[2])
    assert out.weights[1] == pytest.approx(0.25)


def test_all_abstain_ineligible():
    rows = _res([(1, "cdr1", 1.0, -1.0), (2, "cdr1", -1.0, 1.0)])
    out = adaptive_weights(rows, {"cdr1": {"G_drop": 0.01, "G_gain": -0.01}})
    assert out.eligible is False
    assert out.weights == {}
    assert out.diagnostics["n_abstain_regions"] == 1
