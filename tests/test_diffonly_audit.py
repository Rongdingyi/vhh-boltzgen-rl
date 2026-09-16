"""Diff-only multiseed correctness checks (task book §6, §18)."""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/paper_stage/scripts"))

import audit_diffonly_weights as audit  # noqa: E402
import make_diffonly_report as report  # noqa: E402

DESIGN = {1, 2, 3, 4, 5}


def test_uniform_diff_support_passes():
    row = audit.check_pair("p", [1, 2, 4], {"1": 1 / 3, "2": 1 / 3, "4": 1 / 3}, DESIGN)
    assert row["pass"], row
    assert row["n_active"] == row["n_diff"] == 3
    assert row["unique_positive_weights"] == 1


def test_non_uniform_or_incomplete_support_fails():
    row = audit.check_pair("p", [1, 2, 4], {"1": 0.5, "2": 0.5}, DESIGN)
    assert not row["pass"]
    assert row["checks"]["support_is_differing"] is False
    row = audit.check_pair("p", [1, 2, 4], {"1": 0.2, "2": 0.3, "4": 0.5}, DESIGN)
    assert not row["pass"]
    assert row["checks"]["uniform_weights"] is False


def test_weights_on_non_differing_residue_fail():
    row = audit.check_pair("p", [1, 2], {"1": 0.5, "2": 0.25, "3": 0.25}, DESIGN)
    assert not row["pass"]
    assert row["checks"]["no_same_residue_weight"] is False


def test_weights_outside_design_fail():
    row = audit.check_pair("p", [1, 2], {"1": 0.5, "2": 0.25, "9": 0.25}, DESIGN)
    assert not row["pass"]
    assert row["checks"]["no_non_design_weight"] is False


def test_decision_cases():
    assert report.decide([1.0, 0.9, 1.1])["case"] == "C"
    assert report.decide([0.5, 0.4, 0.6])["case"] == "B"
    assert report.decide([0.20, 0.25, 0.35])["case"] == "A"   # mean < +0.30
    assert report.decide([1.0, 0.5, -0.1])["case"] == "A"    # not 3/3
    b = report.decide([0.30, 0.30, 0.30])
    assert b["case"] == "B" and b["three_of_three_positive"] is True


def test_main_table_has_mean_and_std_rows():
    rows = [{"seed": s, "n3_delta": 1.0, "diff_only_delta": 2.0, "cf_delta": 3.0,
             "cf_minus_diff": 1.0, "diff_minus_n3": 1.0} for s in (20260913, 43, 44)]
    table = report._main_table(rows)
    assert table[-2]["seed"] == "mean"
    assert table[-1]["seed"] == "std"
    assert table[-2]["cf_delta"] == pytest.approx(3.0)
    assert table[-1]["cf_minus_diff"] == pytest.approx(0.0)
