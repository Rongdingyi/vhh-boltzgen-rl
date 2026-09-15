"""Pilot wiring regressions: eligible-CF path, eval summary refs, gate guards.

Covers review items 3, 4 and 9 without importing torch or running training.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/adaptive_granularity/scripts"))

import _common as C  # noqa: E402
import eval_pilot  # noqa: E402
import train_pilot  # noqa: E402


def test_eligible_cf_arm_resolves_to_materialized_pairs():
    pairs_path, weights_path, variant = train_pilot.resolve_arm_paths("eligible-cf")
    assert pairs_path.name == "pairs_adaptive_eligible_cf_pilot4.jsonl"
    assert weights_path == C.CURRENT_WEIGHTS
    assert variant == "cf"


def test_simple_arms_resolve_to_variant_names():
    for arm, variant in (("nofloor", "no_floor"), ("strict", "strict_consensus"),
                         ("region", "strict_region"), ("adaptive", "adaptive"),
                         ("shuffle", "adaptive_shuffle")):
        pairs_path, weights_path, got = train_pilot.resolve_arm_paths(arm)
        assert got == variant, arm
        assert pairs_path.name == f"pairs_{variant}_pilot4.jsonl"
        assert weights_path.name == "ag_weights.json"


def test_current_arm_resolves_to_own_materialization():
    pairs_path, weights_path, variant = train_pilot.resolve_arm_paths("current")
    assert variant == "cf"
    assert pairs_path.parent.name == "current"
    assert weights_path.parent.name == "current"


def _summary(rewards: dict[str, float]) -> dict:
    return {"reward_mean": sum(rewards.values()) / len(rewards),
            "cases": {c: {"n": 1, "n_invalid": 0, "n_fr_mismatch": 0,
                          "n_unique": 1, "reward_mean": r}
                      for c, r in rewards.items()}}


def test_eval_summary_handles_per_case_references():
    cases = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    refs = {"base": 1.0, "per_case_base": {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0},
            "current": 2.0, "per_case_current": {"a": 2.0, "b": 3.0, "c": 4.0, "d": 5.0}}
    out = eval_pilot._summarize(_summary(cases), refs)
    assert out["reward_mean"] == pytest.approx(2.5)
    assert out["delta_vs_base"] == pytest.approx(1.5)
    assert out["delta_vs_current"] == pytest.approx(0.5)
    assert out["wins_vs_base"] == "4/0/0"
    assert out["wins_vs_current"] == "0/0/4"


def test_gate_b_guard_blocks_without_override(tmp_path, monkeypatch):
    monkeypatch.setattr(train_pilot, "GATE_B", tmp_path / "gate_b.json")
    with pytest.raises(SystemExit):
        train_pilot.require_gate_b(None)
    (tmp_path / "gate_b.json").write_text(json.dumps({"pass": False}))
    with pytest.raises(SystemExit):
        train_pilot.require_gate_b(None)
    (tmp_path / "gate_b.json").write_text(json.dumps({"pass": True}))
    train_pilot.require_gate_b(None)


def test_gate_a_guard_blocks_without_override(tmp_path, monkeypatch):
    monkeypatch.setattr(train_pilot, "GATE_A", tmp_path / "audit_summary.json")
    with pytest.raises(SystemExit):
        train_pilot.require_gate_a(None)
    (tmp_path / "audit_summary.json").write_text(json.dumps({"gate_a_pass": False}))
    with pytest.raises(SystemExit):
        train_pilot.require_gate_a(None)
    (tmp_path / "audit_summary.json").write_text(json.dumps({"gate_a_pass": True}))
    train_pilot.require_gate_a(None)


def test_gate_b_override_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(train_pilot, "GATE_B", tmp_path / "gate_b.json")
    train_pilot.require_gate_b("manual smoke only")   # must not raise


def test_gate_c_guard_requires_strong_go(tmp_path, monkeypatch):
    import train_full
    monkeypatch.setattr(train_full, "GATE_C", tmp_path / "gate_c.json")
    with pytest.raises(SystemExit):
        train_full.require_gate_c(None)
    (tmp_path / "gate_c.json").write_text(json.dumps({"verdict": "BORDERLINE_3SEED"}))
    with pytest.raises(SystemExit):
        train_full.require_gate_c(None)
    (tmp_path / "gate_c.json").write_text(json.dumps({"verdict": "STRONG_GO"}))
    train_full.require_gate_c(None)
