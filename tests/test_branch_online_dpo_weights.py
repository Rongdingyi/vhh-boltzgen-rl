"""Arm B uniform changed-position weights; no credit dependency (§44/§45)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.branch_distill.online_diff_dpo import uniform_changed_weights  # noqa: E402


def test_uniform_weights_over_changed_positions():
    weights = uniform_changed_weights((5, 2, 9))
    assert sorted(weights) == [2, 5, 9]
    assert all(v == pytest.approx(1 / 3) for v in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)


def test_empty_changed_positions_rejected():
    with pytest.raises(ValueError):
        uniform_changed_weights(())
