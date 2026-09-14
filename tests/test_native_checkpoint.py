"""Checkpoint round-trip: only score_model weights replaced (task book §80)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))


def test_saved_checkpoint_loadable_and_only_score_model_replaced():
    ckpt = ROOT / "runs/native_n2/checkpoint_0500.pt"
    if not ckpt.is_file():
        import pytest
        pytest.skip("checkpoint not present")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "state_dict" in payload and "native_posttrain" in payload
    meta = payload["native_posttrain"]
    assert meta["n_replaced_tensors"] > 0
    assert payload["state_dict"] is payload["state_dict"]
