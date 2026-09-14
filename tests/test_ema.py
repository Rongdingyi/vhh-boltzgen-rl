"""EMA math (task book §82)."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cf_opsd.behavior_ema import ema_update


class _M(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor(v))


def test_ema_math():
    b, s = _M(1.0), _M(3.0)
    ema_update(b, s, decay=0.99)
    assert abs(float(b.p) - (0.99 * 1.0 + 0.01 * 3.0)) < 1e-6
