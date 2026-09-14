"""DPO init log(2) gate (task book §38/§66)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vhh_rl.native_atom14.dpo_loss import diffusion_dpo_loss  # noqa: E402


def test_init_loss_is_log2():
    torch.manual_seed(0)
    l = torch.rand(8)
    out = diffusion_dpo_loss(l, l, l, l, beta=10.0)
    assert abs(float(out.total_loss) - math.log(2)) < 1e-6
