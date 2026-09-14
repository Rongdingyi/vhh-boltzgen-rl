"""Only score_model parameters receive gradients (task book §82)."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest
import torch

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))


def test_only_score_model_params():
    pytest.importorskip("boltzgen")
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    from vhh_rl.cf_opsd.checkpoint import load_base_model, trainable_score_params

    base = load_base_model(
        Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt"),
        device="cuda")
    params = trainable_score_params(base)
    names = [n for n, _ in base.named_parameters() if _.requires_grad]
    assert names, "no trainable params"
    assert all(n.startswith("structure_module.score_model") for n in names), names[:3]
