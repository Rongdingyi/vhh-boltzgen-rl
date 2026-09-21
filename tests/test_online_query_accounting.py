"""Reward scorer queries vs pair records (task book §56 + P0 fix)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from _online_fixtures import make_group  # noqa: E402
from vhh_rl.online_pref.metrics import (  # noqa: E402
    generated_endpoint_count, reward_query_count,
)


def test_reward_queries_count_scored_siblings_only():
    group = make_group()
    group.siblings[1].reward = None
    assert reward_query_count([group]) == 2
    assert generated_endpoint_count([group]) == 3


def test_group_stats_and_zero_case():
    assert reward_query_count([]) == 0


def test_branch_distill_round_log_has_corrected_fields():
    """P0 regression: the old single 'queries' field was a pair-record count."""
    src = (Path(__file__).resolve().parents[1]
           / "experiments/branch_distill/scripts/gate3_train.py").read_text()
    assert "reward_queries_round" in src
    assert "pair_records_round" in src
    report = (Path(__file__).resolve().parents[1]
              / "experiments/branch_distill/scripts/gate3_report.py").read_text()
    assert "reward_queries_cumulative" in report
