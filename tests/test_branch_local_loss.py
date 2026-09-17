"""Masked local distillation loss (task book §36)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.branch_distill.query_fit import local_distill_loss  # noqa: E402


def test_loss_only_reads_masked_atoms():
    pred = torch.zeros(6, 3)
    target = torch.ones(6, 3)
    mask = torch.zeros(6, dtype=torch.bool)
    mask[[1, 4]] = True
    loss = local_distill_loss(pred, target, mask)
    assert float(loss) == pytest.approx(3.0)         # 3 coords each, unit diff


def test_empty_mask_raises():
    with pytest.raises(ValueError):
        local_distill_loss(torch.zeros(4, 3), torch.zeros(4, 3),
                           torch.zeros(4, dtype=torch.bool))
