"""Edge orientation + no-reward-magnitude-leakage tests (task book §73/§74/§82)."""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.signed_local.types import (  # noqa: E402
    TOL, LocalPreferenceEdge, classify_event, preferred_side, robust_sign,
)


def _edge(anchor_reward: float, cf_reward: float, context: str,
          c_drop: float = 0.0, c_gain: float = 0.0) -> LocalPreferenceEdge:
    dR = cf_reward - anchor_reward
    return LocalPreferenceEdge(
        edge_id="c:w:l:%s:1" % context, case_id="c", pair_id="c:w:l", position=1,
        context=context, event_class="sign_flip", anchor_sample_id="w",
        donor_sample_id="l", anchor_sequence="AA", cf_sequence="AB",
        anchor_reward=anchor_reward, cf_reward=cf_reward, dR=dR,
        preferred_side=preferred_side(dR), anchor_coords_path="a.pt",
        cf_coords_path="b.pt", c_drop=c_drop, c_gain=c_gain,
        touched_atoms=10, moved_rms=0.5, invalid=False, fr_mismatch=0)


def test_preferred_side_orientation():
    assert preferred_side(+2.0) == "cf"
    assert preferred_side(-2.0) == "anchor"
    assert preferred_side(+0.01) == "skip"
    e = _edge(5.0, 7.0, "winner_drop", c_drop=-2.0)
    assert e.preferred_side == "cf"
    e = _edge(7.0, 5.0, "winner_drop", c_drop=+2.0)
    assert e.preferred_side == "anchor"


def test_drop_gain_dR_invariants():
    e = _edge(5.0, 3.0, "winner_drop", c_drop=+2.0)   # dR = -2 = -c_drop
    e.validate()
    e = _edge(5.0, 2.0, "loser_gain", c_gain=-3.0)    # dR = -3 = +c_gain
    e.validate()
    with pytest.raises(ValueError):
        _edge(5.0, 3.0, "winner_drop", c_drop=-2.0).validate()
    with pytest.raises(ValueError):
        _edge(5.0, 2.0, "loser_gain", c_gain=+3.0).validate()


def test_classify_tolerance():
    cases = {
        (1, 1): "both_positive", (-1, -1): "both_negative",
        (1, -1): "sign_flip", (-1, 1): "sign_flip",
        (0, -1): "weak", (1, 0): "weak",
    }
    for (a, b), expected in cases.items():
        assert classify_event(a, b) == expected
    assert robust_sign(0.01) == 0 and robust_sign(-0.01) == 0
    assert robust_sign(0.06) == 1 and robust_sign(-0.06) == -1


def test_no_reward_magnitude_leakage_in_training_code():
    """§82: dR may appear only for orientation/threshold/logging."""
    forbidden = ("sigmoid(dR", "dR/tau", "kappa", "reward_target", "delta_r / tau")
    for rel in ("src/vhh_rl/signed_local/trainer.py",
                "src/vhh_rl/signed_local/local_dpo.py",
                "src/vhh_rl/native_atom14/global_cf_step.py"):
        text = (ROOT / rel).read_text()
        for token in forbidden:
            assert token not in text, f"{token} found in {rel}"
