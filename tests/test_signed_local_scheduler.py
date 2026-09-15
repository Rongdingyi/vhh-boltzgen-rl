"""Deterministic G-G-G-L schedule (task book §34/§69/§80)."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.signed_local.scheduler import ThreeToOneScheduler  # noqa: E402


def test_branch_pattern_and_counts():
    s = ThreeToOneScheduler(3, 1)
    pattern = [s.branch(i) for i in range(1, 9)]
    assert pattern == ["global", "global", "global", "local",
                       "global", "global", "global", "local"]
    counts = s.counts(100)
    assert counts == {"global": 75, "local": 25}


def test_bad_step_and_sizes():
    s = ThreeToOneScheduler(3, 1)
    try:
        s.branch(0)
        raise AssertionError("step 0 must raise")
    except ValueError:
        pass
    assert ThreeToOneScheduler(1, 1).counts(10) == {"global": 5, "local": 5}
