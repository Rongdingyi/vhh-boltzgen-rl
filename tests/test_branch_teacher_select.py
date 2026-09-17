"""Teacher/peer selection and eligibility (task book §28/§43)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.branch_distill.teacher_select import (  # noqa: E402
    eligible, rank_valid, select_teacher_peer, valid_siblings,
)
from vhh_rl.branch_distill.types import SiblingEndpoint  # noqa: E402


def _sib(index, reward, seq="AAAA", invalid=False, fr=0):
    return SiblingEndpoint(branch_index=index, endpoint_coords=torch.zeros(5, 3),
                           endpoint_sequence=seq, reward=reward,
                           contains_invalid=invalid, fr_mismatch=fr,
                           query_coords=torch.zeros(5, 3),
                           anchor_coords=torch.zeros(5, 3))


def test_teacher_is_best_valid_and_peers_selectable():
    siblings = [_sib(0, 1.0), _sib(1, 5.0), _sib(2, 3.0), _sib(3, None),
                _sib(4, 9.9, invalid=True), _sib(5, 8.8, fr=1)]
    valid = valid_siblings(siblings)
    assert [s.branch_index for s in valid] == [0, 1, 2]
    ranked = rank_valid(siblings)
    assert ranked[0].branch_index == 1
    changed = lambda t, p: (1,) if t != p else ()   # noqa: E731
    sel_worst = select_teacher_peer(siblings, "worst", changed)
    assert sel_worst["teacher_index"] == 1 and sel_worst["peer_index"] == 0
    assert sel_worst["reward_gap"] == 4.0
    sel_median = select_teacher_peer(siblings, "median", changed)
    assert sel_median["peer_index"] == 2


def test_eligible_requires_gap_and_changed_position():
    selection = {"teacher_index": 1, "peer_index": 0, "teacher_reward": 5.0,
                 "peer_reward": 4.5, "reward_gap": 0.5, "changed_positions": (2,),
                 "teacher_sequence": "AA", "peer_sequence": "AB"}
    assert eligible(selection, 0.30)
    shallow = dict(selection, reward_gap=0.2)
    assert not eligible(shallow, 0.30)
    no_change = dict(selection, changed_positions=())
    assert not eligible(no_change, 0.30)
    assert not eligible(None, 0.30)
