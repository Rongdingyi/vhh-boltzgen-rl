"""Gate thresholds, verdicts and protocol-hash guard (task book §26/§39/§54/§61)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.branch_distill import gates  # noqa: E402


def _group(valid_rate=1.0, unique=4, std=0.3, bmm=0.5, tvm=2):
    return {"valid_rate": valid_rate, "n_unique_valid_sequences": unique,
            "reward_std": std, "best_minus_median": bmm,
            "teacher_vs_median_hamming": tvm}


def test_gate1_group_thresholds():
    assert gates.gate1_group_pass(_group())
    assert not gates.gate1_group_pass(_group(valid_rate=0.5))
    assert not gates.gate1_group_pass(_group(unique=2))
    assert not gates.gate1_group_pass(_group(std=0.1))
    assert not gates.gate1_group_pass(_group(bmm=0.1))
    assert not gates.gate1_group_pass(_group(tvm=0))


def test_gate1_progress_requires_six_of_eight():
    six = [_group() for _ in range(6)] + [_group(valid_rate=0.2) for _ in range(2)]
    verdict = gates.gate1_progress_pass(six)
    assert verdict["pass"] and verdict["n_groups_pass"] == 6
    five = [_group() for _ in range(5)] + [_group(valid_rate=0.2) for _ in range(3)]
    assert not gates.gate1_progress_pass(five)["pass"]


def _gate2_row(replay=1e-6, invalid=False, fr=0, match=True, ratio=0.3,
               before=3, after=1, postfit=1.0, peer=0.9):
    return {"replay_maxdiff": replay, "carrier_invalid": invalid, "carrier_fr": fr,
            "carrier_all_match": match, "mse_ratio": ratio,
            "hamming_before": before, "hamming_after": after,
            "postfit_reward": postfit, "peer_reward": peer}


def test_gate2_all_checks_and_verdict():
    rows = [_gate2_row() for _ in range(12)]
    verdict = gates.gate2_verdict(rows)
    assert verdict["pass"], verdict
    bad_replay = [dict(r, replay_maxdiff=1e-3) for r in rows]
    assert not gates.gate2_verdict(bad_replay)["A_replay"]
    bad_target = [dict(r, carrier_all_match=False) for r in rows]
    assert not gates.gate2_verdict(bad_target)["B_target"]
    bad_fit = [dict(r, mse_ratio=0.9) for r in rows]
    assert not gates.gate2_verdict(bad_fit)["C_learnability"]
    bad_down = [dict(r, hamming_after=5) for r in rows]
    assert not gates.gate2_verdict(bad_down)["D_downstream"]
    bad_safe = [dict(r, postfit_reward=0.1, peer_reward=1.0) for r in rows]
    assert not gates.gate2_verdict(bad_safe)["E_safety"]


def test_gate3_verdicts():
    per_case = {f"case{i}": {"A": 1.0, "D": 1.6} for i in range(8)}
    strong = gates.gate3_verdict({"A": 1.0, "B": 1.05, "C": 1.05, "D": 1.6},
                                 per_case, invalid_rate=0.0, fr_mismatch=0)
    assert strong["verdict"] == "STRONG_GO"
    no_go = gates.gate3_verdict({"A": 1.5, "B": 1.6, "C": 1.6, "D": 1.2},
                                per_case, invalid_rate=0.0, fr_mismatch=0)
    assert no_go["verdict"] == "NO_GO"
    weak = gates.gate3_verdict({"A": 1.0, "B": 1.05, "C": 1.05, "D": 1.3},
                               per_case, invalid_rate=0.0, fr_mismatch=0)
    assert weak["verdict"] == "INCONCLUSIVE"


def test_protocol_hash_guard(tmp_path):
    frozen = tmp_path / "FROZEN_PROTOCOL.yaml"
    frozen.write_text("k: v\n")
    digest = hashlib.sha256(b"k: v\n").hexdigest()
    gates.guard_protocol({"protocol_sha256": digest}, frozen)
    with pytest.raises(RuntimeError):
        gates.guard_protocol({"protocol_sha256": "deadbeef"}, frozen)
