"""Phase E refuses to run unless Gates B/C/D are all satisfied (audit fix C)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cf_opsd.onpolicy_trainer import check_gates, run_onpolicy  # noqa: E402


def _write_gates(tmp_path: Path, *, radius, gate_c_pass, gate_d_pass) -> dict:
    gb = tmp_path / "gate_b.json"
    gb.write_text(json.dumps({"selected_radius": radius}))
    gc = tmp_path / "gate_c.json"
    gc.write_text(json.dumps({"gate_c": {"V0-A": gate_c_pass, "V0-B": False}}))
    gd = tmp_path / "gate_d.json"
    gd.write_text(json.dumps({"gate_d": {"50": {"reward_superiority": gate_d_pass}}}))
    return {"gate_b": gb, "gate_c": gc, "gate_d": gd}


def test_check_gates_pass(tmp_path):
    g = _write_gates(tmp_path, radius=0.5, gate_c_pass=True, gate_d_pass=True)
    check_gates(g["gate_b"], g["gate_c"], g["gate_d"])  # no raise


@pytest.mark.parametrize("radius,c,d", [(None, True, True), (0.5, False, True), (0.5, True, False)])
def test_check_gates_refuse(tmp_path, radius, c, d):
    g = _write_gates(tmp_path, radius=radius, gate_c_pass=c, gate_d_pass=d)
    with pytest.raises(RuntimeError):
        check_gates(g["gate_b"], g["gate_c"], g["gate_d"])


def test_run_onpolicy_refuses_before_model_work(tmp_path):
    """Gate failure must raise before any dependency is used."""
    case_cfg = yaml.safe_load((ROOT / "configs/cf_opsd/fixed_cases.yaml").read_text())
    loop_cfg = yaml.safe_load((ROOT / "configs/cf_opsd/onpolicy_v1.yaml").read_text())
    case_path = tmp_path / "fixed.yaml"
    loop_path = tmp_path / "loop.yaml"
    case_path.write_text(yaml.safe_dump(case_cfg))
    loop_path.write_text(yaml.safe_dump(loop_cfg))
    qp = tmp_path / "q.json"
    qp.write_text(json.dumps({"q_star_progress": 0.9}))
    g = _write_gates(tmp_path, radius=None, gate_c_pass=True, gate_d_pass=True)

    called = {"n": 0}

    def boom(*a, **k):
        called["n"] += 1
        raise AssertionError("dependencies must not be used before gate checks")

    with pytest.raises(RuntimeError):
        run_onpolicy(base_checkpoint=tmp_path / "none.pt", case_config=case_path,
                     loop_config=loop_path, query_probe=qp,
                     gate_b_path=g["gate_b"], gate_c_path=g["gate_c"],
                     gate_d_path=g["gate_d"], output_dir=tmp_path / "out",
                     deps={"load_base": boom, "adapter_factory": boom})
    assert called["n"] == 0
