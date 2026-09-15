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


def test_current_arm_resolves_to_weight_inputs():
    pairs_path, weights_path, variant = train_pilot.resolve_arm_paths("current")
    assert variant == "cf"
    assert pairs_path.parent == C.WEIGHTS_DIR
    assert weights_path.parent == C.WEIGHTS_DIR
    assert pairs_path.name == "pairs_current_cf_pilot4.jsonl"
    assert weights_path.name == "current_cf_pilot4_weights.json"


def test_current_pilot_inputs_survive_output_cleanup(tmp_path, monkeypatch):
    """P0 regression: materialize -> rmtree(output dir) -> trainer reopen.

    Inputs must live outside the training output directory.
    """
    import shutil

    weights_dir = tmp_path / "weights"
    pilot_dir = tmp_path / "pilot"
    monkeypatch.setattr(C, "WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr(C, "PILOT_DIR", pilot_dir)
    monkeypatch.setattr(C, "PILOT_TRAIN_CASES", ["case_a"])
    monkeypatch.setattr(C, "load_current_weights", lambda: {
        "eta": 0.75, "seed": 1,
        "pairs": {"case_a:w:l": {"case_id": "case_a", "cf": {"10": 1.0}},
                  "case_b:w:l": {"case_id": "case_b", "cf": {"11": 1.0}}},
    })
    pairs = [{"case_id": "case_a", "winner_sample_id": "w", "loser_sample_id": "l"},
             {"case_id": "case_b", "winner_sample_id": "w", "loser_sample_id": "l"}]

    pairs_path, weights_path = train_pilot._materialize_current_cf(pairs)
    assert pairs_path == C.WEIGHTS_DIR / "pairs_current_cf_pilot4.jsonl"
    assert weights_path == C.WEIGHTS_DIR / "current_cf_pilot4_weights.json"
    assert pairs_path.is_file() and weights_path.is_file()

    out_dir = train_pilot.arm_output_dir("current")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_metrics.jsonl").write_text("old\n")
    shutil.rmtree(out_dir, ignore_errors=True)      # production cleanup

    assert pairs_path.is_file(), "current-CF pairs input was deleted by cleanup"
    assert weights_path.is_file(), "current-CF weights input was deleted by cleanup"
    rows = [json.loads(l) for l in pairs_path.open()]
    assert {r["case_id"] for r in rows} == {"case_a"}
    weights = json.loads(weights_path.read_text())
    assert set(weights["pairs"]) == {"case_a:w:l"}


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


def test_seed_specific_output_dirs_do_not_collide():
    primary = train_pilot.arm_output_dir("current")
    seed42 = train_pilot.arm_output_dir("current", 42)
    assert primary != seed42
    assert seed42.parent.name == "seed42"
    assert train_pilot.arm_output_dir("adaptive", 43).name == "adaptive"
    assert train_pilot.arm_output_dir("adaptive", 43).parent.name == "seed43"


def test_three_seed_confirm_rule():
    import eval_3seed
    go = eval_3seed.confirm_verdict({42: 0.40, 43: 0.35, 44: 0.20})
    assert go["verdict"] == "GO"                          # mean 0.317, 3/3
    assert go["seeds_adaptive_ge_cf"] == 3
    two_of_three = eval_3seed.confirm_verdict({42: 0.9, 43: 0.1, 44: -0.05})
    assert two_of_three["verdict"] == "GO"                # mean 0.317, 2/3
    assert two_of_three["seeds_adaptive_ge_cf"] == 2
    marginal = eval_3seed.confirm_verdict({42: 0.25, 43: 0.20, 44: 0.30})
    assert marginal["verdict"] == "NO_GO"                 # mean < +0.30
    one_positive = eval_3seed.confirm_verdict({42: 1.5, 43: -0.5, 44: -0.1})
    assert one_positive["verdict"] == "NO_GO"             # mean ok, 1/3 non-negative
    assert eval_3seed.confirm_verdict({})["verdict"] == "NOT_RUN"


def test_full_multiseed_guard_requires_full_go(tmp_path, monkeypatch):
    import train_full
    monkeypatch.setattr(train_full, "GATE_FULL", tmp_path / "gate_full.json")
    with pytest.raises(SystemExit):
        train_full.require_gate_full(None)
    (tmp_path / "gate_full.json").write_text(json.dumps({"verdict": "NOT_PASSED"}))
    with pytest.raises(SystemExit):
        train_full.require_gate_full(None)
    (tmp_path / "gate_full.json").write_text(json.dumps({"verdict": "FULL_GO"}))
    train_full.require_gate_full(None)


def test_full_report_gate_names(tmp_path, monkeypatch):
    import make_full_report
    base_cases = {f"case_{i}": 0.0 for i in range(8)}   # no manifest read in CI
    payload = {
        "base": {"reward_mean": 0.0},
        "f0_seed20260913": {"selected": {"reward_mean": 1.0, "per_case": base_cases,
                                         "invalid": 0, "valid": True}},
        "f2_seed20260913": {"selected": {"reward_mean": 1.6, "per_case": base_cases,
                                         "invalid": 0, "valid": True}},
        "f3_seed20260913": {"selected": {"reward_mean": 0.5, "per_case": base_cases,
                                         "invalid": 0, "valid": True}},
        "f4_seed20260913": {"selected": {"reward_mean": 0.5, "per_case": base_cases,
                                         "invalid": 0, "valid": True}},
    }
    assert make_full_report._gate(payload)["verdict"] == "FULL_GO"
    assert make_full_report._gate_multiseed(payload)["verdict"] == "NOT_RUN"
    for seed in (42, 43, 44):
        payload[f"f0_seed{seed}"] = {"selected": {"reward_mean": 1.0}}
        payload[f"f2_seed{seed}"] = {"selected": {"reward_mean": 1.6}}
    multi = make_full_report._gate_multiseed(payload)
    assert multi["verdict"] == "MULTISEED_GO"
    assert multi["seeds_non_negative"] == 3


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
