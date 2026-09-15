"""V2 Strict-consensus CF (task book §20/§27)."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.credit.adaptive_granularity import strict_consensus_weights  # noqa: E402


def _rows(pairs_):
    return [{"position": 20 + i, "region": "cdr2", "c_drop": d, "c_gain": g,
             "c_cons": min(d, g) if d > 0 and g > 0 else 0.0}
            for i, (d, g) in enumerate(pairs_)]


def test_strict_consensus_keeps_only_robust_stable_positive():
    out = strict_consensus_weights(_rows([(1.0, 2.0), (0.04, 1.0),
                                          (-1.0, -2.0), (1.0, -1.0)]))
    assert out.eligible
    assert out.weights == {20: 1.0}
    assert out.diagnostics["n_stable_positive"] == 1


def test_strict_consensus_weights_are_min_of_margins():
    out = strict_consensus_weights(_rows([(3.0, 1.5), (2.0, 2.0)]))
    assert out.weights == {20: 1.5 / 3.5, 21: 2.0 / 3.5}


def test_strict_consensus_all_weak_ineligible():
    out = strict_consensus_weights(_rows([(0.04, 1.0), (0.0, 0.0), (1.0, -1.0)]))
    assert out.eligible is False
    assert out.weights == {}
