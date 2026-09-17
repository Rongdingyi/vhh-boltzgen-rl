"""Changed positions must only compare design positions (task book §19)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.branch_distill.local_target import changed_design_positions  # noqa: E402


def test_only_design_positions_are_compared():
    teacher = "AAAAAA"
    peer = "ABABAB"
    assert changed_design_positions(teacher, peer, (1, 3, 5)) == (1, 3, 5)
    assert changed_design_positions(teacher, peer, (2, 4)) == ()


def test_fr_style_constant_positions_ignored():
    teacher = "MKA"
    peer = "MKB"
    assert changed_design_positions(teacher, peer, (2,)) == (2,)
    assert changed_design_positions(teacher, peer, (0, 1)) == ()
    assert changed_design_positions(teacher, peer, (0, 1, 2)) == (2,)
