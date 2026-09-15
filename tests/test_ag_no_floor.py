"""V1 No-floor CF (task book §19/§25/§26)."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.credit.adaptive_granularity import no_floor_weights  # noqa: E402


def _rows(cons, drop=None, gain=None):
    drop = drop or [1.0] * len(cons)
    gain = gain or [1.0] * len(cons)
    return [{"position": 10 + i, "region": "cdr1", "c_drop": drop[i],
             "c_gain": gain[i], "c_cons": cons[i]} for i in range(len(cons))]


def test_no_floor_normalizes_cons_credit():
    out = no_floor_weights(_rows([2.0, 1.0, 0.0, 0.0]))
    assert out.eligible
    assert out.weights == {10: 2.0 / 3.0, 11: 1.0 / 3.0}


def test_no_floor_ignores_negative_and_weak_cons():
    out = no_floor_weights(_rows([-5.0, 0.5, 0.0]))
    assert out.eligible
    assert out.weights == {11: 1.0}


def test_no_floor_empty_is_ineligible_not_uniform():
    out = no_floor_weights(_rows([0.0, 0.0, 0.0]))
    assert out.eligible is False
    assert out.weights == {}
    assert "sum_c_cons" in out.reason
