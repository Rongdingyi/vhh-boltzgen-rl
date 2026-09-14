"""On-policy config completeness (task book §86; audit fix for KeyError)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cf_opsd.onpolicy_trainer import REQUIRED_KEYS, _require  # noqa: E402


def merged_config() -> dict:
    case_cfg = yaml.safe_load((ROOT / "configs/cf_opsd/fixed_cases.yaml").read_text())
    loop_cfg = yaml.safe_load((ROOT / "configs/cf_opsd/onpolicy_v1.yaml").read_text())
    return {**case_cfg, **loop_cfg}


def test_onpolicy_config_has_required_sections():
    cfg = merged_config()
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    assert not missing, f"missing sections: {missing}"
    assert cfg["outer"]["rounds"] >= 1
    assert cfg["fit"]["updates_per_round"] >= 1
    assert cfg["optimizer"]["lr"] > 0
    assert cfg["train_cases"] and cfg["heldout_cases"]


def test_require_raises_on_missing():
    with pytest.raises(KeyError) as exc:
        _require({"train_cases": []})
    assert "outer" in str(exc.value)
