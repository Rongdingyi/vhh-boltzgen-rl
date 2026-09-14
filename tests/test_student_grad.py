"""Only score_model parameters receive gradients (task book §82)."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_only_score_model_params():
    pytest.importorskip("boltzgen")
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    from vhh_rl.cf_opsd.checkpoint import (
        load_base_model, make_policy_reference, trainable_score_params,
    )

    base = load_base_model(
        Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt"),
        device="cuda")
    policy, reference = make_policy_reference(base, device="cuda")
    names = [n for n, p in policy.named_parameters() if p.requires_grad]
    assert names, "no trainable params"
    assert all(n.startswith("structure_module.score_model") for n in names), names[:3]
    assert trainable_score_params(policy)
    assert not any(p.requires_grad for p in reference.parameters()), "reference must be frozen"
