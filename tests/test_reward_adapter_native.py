"""Reward adapter direction/field (task book §2)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))


def test_worker_field_is_cdr_margin():
    src = (ROOT / "src/vhh_rl/rewards/scorer_worker.py").read_text()
    assert "cdr_margin_guarded" in src
    assert "result.cdr_camelid_margin" in src


def test_adapter_direction_higher():
    from vhh_rl.rewards.scoring import ScorerAdapter
    assert ScorerAdapter.direction_is_higher(object()) is True
