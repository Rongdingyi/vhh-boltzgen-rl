"""Local target support + carrier decode contract (task book §29/§30/§58)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from _branch_fixtures import make_coords, make_feats  # noqa: E402
from vhh_rl.branch_distill.local_target import (  # noqa: E402
    build_target, carrier_positions_match, target_mask,
)


def test_target_touches_only_changed_fake_atoms():
    feats = make_feats()
    peer_anchor = make_coords(seed=1)
    teacher = make_coords(seed=2, shift_fake=1.5)
    target, touched = build_target(peer_anchor, teacher, feats, (2,))
    assert touched.sum() == 2                       # fake atoms 11,12
    assert set(torch.where(touched)[0].tolist()) == {11, 12}
    # unchanged atoms are byte-identical
    unchanged = torch.ones(17, dtype=torch.bool)
    unchanged[[11, 12]] = False
    assert torch.equal(target[unchanged], peer_anchor[unchanged])
    # moved atoms are not identical
    assert not torch.allclose(target[[11, 12]], peer_anchor[[11, 12]])


def test_target_mask_is_subset_of_fake_and_changed():
    feats = make_feats()
    peer_anchor = make_coords(seed=3)
    teacher = make_coords(seed=4, shift_fake=2.0)
    _target, touched = build_target(peer_anchor, teacher, feats, (2,))
    mask = target_mask(touched, feats)
    assert set(torch.where(mask)[0].tolist()) == {11, 12}
    token_of_atom = feats["atom_to_token"].argmax(-1)
    assert set(token_of_atom[mask].tolist()) == {2}
    assert bool((mask & ~feats["fake_atom_mask"]).sum() == 0)


def test_carrier_position_match_helper():
    matches = carrier_positions_match("MKA", "MKB", (0, 1, 2))
    assert matches == {0: True, 1: True, 2: False}


def test_verified_changed_positions_keeps_only_matching_positions():
    """Amendment 1: target supervision uses the carrier-verified subset."""
    from vhh_rl.branch_distill.local_target import verified_changed_positions

    matches = {24: True, 25: False, 26: False, 29: True}
    assert verified_changed_positions(matches, (24, 25, 26, 29)) == (24, 29)
    assert verified_changed_positions({}, (1, 2)) == ()
