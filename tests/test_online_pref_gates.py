"""Phase gates and protocol hash guard (task book §14/§48)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.online_pref import gates  # noqa: E402


def test_phase0_gate_thresholds():
    ok = gates.phase0_verdict({1: 0.5, 2: 0.4, 3: 0.05}, {1: 5, 2: 6, 3: 4},
                              invalid_rate=0.0, fr_mismatch=0)
    assert ok["pass"] and ok["seeds_positive"] == 3 and ok["median_cases_ge"] == 5
    two_positive = gates.phase0_verdict({1: 0.9, 2: 0.5, 3: -0.1}, {1: 5, 2: 6, 3: 1},
                                        invalid_rate=0.0, fr_mismatch=0)
    assert two_positive["seeds_positive"] == 2 and two_positive["pass"]
    weak_mean = gates.phase0_verdict({1: 0.2, 2: 0.1, 3: 0.3}, {1: 5, 2: 5, 3: 5},
                                     0.0, 0)
    assert not weak_mean["pass"]
    one_seed = gates.phase0_verdict({1: 0.9, 2: -0.1, 3: -0.2}, {1: 5, 2: 5, 3: 5},
                                    0.0, 0)
    assert not one_seed["pass"]
    many_cases = gates.phase0_verdict({1: 0.9, 2: 0.9, 3: 0.9}, {1: 1, 2: 1, 3: 1},
                                      0.0, 0)
    assert not many_cases["pass"]


def test_invalid_and_fr_block_every_gate():
    assert not gates.phase0_verdict({1: 1.0}, {1: 8}, invalid_rate=0.02,
                                    fr_mismatch=0)["pass"]
    assert not gates.phase0_verdict({1: 1.0}, {1: 8}, invalid_rate=0.0,
                                    fr_mismatch=1)["pass"]


def test_seed_mean_uses_sample_std():
    mean, std = gates.signed_seed_mean_std([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert abs(std - 1.0) < 1e-12        # stdev, not pstdev


def test_protocol_guard(tmp_path):
    frozen = tmp_path / "FROZEN_PROTOCOL.yaml"
    frozen.write_text("k: v\n")
    digest = gates.protocol_hash(frozen)
    gates.guard_protocol({"protocol_sha256": digest}, frozen)
    with pytest.raises(RuntimeError):
        gates.guard_protocol({"protocol_sha256": "nope"}, frozen)
