"""AG robust residue/region classification (task book §9/§10/§73)."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.credit.adaptive_granularity import (  # noqa: E402
    REGION_FLIP, REGION_STABLE_NEGATIVE, REGION_STABLE_POSITIVE, REGION_WEAK,
    SIGN_FLIP, STABLE_NEGATIVE, STABLE_POSITIVE, TOL, WEAK,
    classify_region, classify_residue, robust_sign,
)


def test_robust_sign_uses_tolerance_band():
    assert robust_sign(0.06) == 1
    assert robust_sign(-0.06) == -1
    assert robust_sign(0.05) == 0
    assert robust_sign(-0.05) == 0
    assert robust_sign(1e-12) == 0
    assert robust_sign(float("nan")) == 0
    assert robust_sign(None) == 0


def test_residue_classes():
    assert classify_residue(1.0, 2.0) == STABLE_POSITIVE
    assert classify_residue(-1.0, -2.0) == STABLE_NEGATIVE
    assert classify_residue(1.0, -1.0) == SIGN_FLIP
    assert classify_residue(-1.0, 1.0) == SIGN_FLIP
    assert classify_residue(0.04, 1.0) == WEAK
    assert classify_residue(1.0, -0.04) == WEAK
    assert classify_residue(1e-9, -1e-9) == WEAK  # tiny values never flip


def test_region_classes():
    assert classify_region(0.06, 1.0) == REGION_STABLE_POSITIVE
    assert classify_region(-1.0, -0.06) == REGION_STABLE_NEGATIVE
    assert classify_region(1.0, -1.0) == REGION_FLIP
    assert classify_region(0.2, 0.01) == REGION_WEAK


def test_tol_constant_is_preregistered():
    assert TOL == 0.05
