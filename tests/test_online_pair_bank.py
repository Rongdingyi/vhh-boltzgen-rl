"""Pair banks keep full sibling groups and per-round metadata (§8)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from _online_fixtures import make_group  # noqa: E402
from vhh_rl.online_pref.pair_bank import load_pair_bank, save_pair_bank  # noqa: E402
from vhh_rl.online_pref.pair_builder import build_all_changed_pairs  # noqa: E402


def test_round_trip_keeps_groups_and_pairs(tmp_path):
    group = make_group()
    pairs = build_all_changed_pairs(group, (1, 2, 3))
    path = tmp_path / "pair_bank_r1.pt"
    save_pair_bank(path, round_index=1, behavior_checkpoint_sha256="abc",
                   branch_progress=0.60, groups=[group], pairs=pairs,
                   reward_queries=3)
    payload = load_pair_bank(path)
    assert payload["round"] == 1
    assert payload["behavior_checkpoint_sha256"] == "abc"
    assert payload["reward_queries"] == 3
    assert len(payload["groups"]) == 1
    assert len(payload["groups"][0]["siblings"]) == 3
    assert [p.pair_id for p in payload["pairs"]] == [p.pair_id for p in pairs]
