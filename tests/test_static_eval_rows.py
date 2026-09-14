"""Regression test for cf_opsd_static_eval row/step construction (audit fix)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
SCRIPT = ROOT / "scripts/cf_opsd_static_eval.py"


def _load():
    spec = importlib.util.spec_from_file_location("cf_opsd_static_eval", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cf_opsd_static_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_rows_has_step_and_gate():
    mod = _load()
    results = {
        "base": {"reward_mean": 1.0},
        "cf_dpo_mini_u50": {"reward_mean": 2.0},
        "cf_dpo_mini_u100": {"reward_mean": 3.0},
        "cf_opsd_static_u50": {"reward_mean": 2.2},
        "cf_opsd_static_u100": {"reward_mean": 2.5},
    }
    rows = mod.build_rows(results)
    assert all("step" in r for r in rows)
    assert {r["step"] for r in rows} == {50, 100}
    gate = mod.gate_d_rows(rows)
    import pytest
    assert gate[50]["opsd_delta"] == pytest.approx(1.2)
    assert gate[50]["cfdpo_delta"] == pytest.approx(1.0)
    assert gate[50]["reward_superiority"] is True       # 1.2 >= 1.10 * 1.0
    assert gate[100]["reward_superiority"] is False     # 1.5 < 1.10 * 2.0


def test_build_rows_handles_missing_base():
    mod = _load()
    rows = mod.build_rows({"cf_opsd_static_u50": {"reward_mean": 2.0}})
    assert rows[0]["delta_vs_base"] is None
