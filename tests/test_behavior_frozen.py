"""Optimizer step on student leaves behavior unchanged (task book §82)."""
from __future__ import annotations
import hashlib
import sys
from pathlib import Path
import torch

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))


class _M(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor(v))


def _hash(m):
    return hashlib.sha1(str([round(float(x), 6) for x in m.state_dict()["p"].flatten()]).encode()).hexdigest()


def test_behavior_frozen():
    student, behavior = _M(1.0), _M(1.0)
    h0 = _hash(behavior)
    opt = torch.optim.SGD(student.parameters(), lr=0.1)
    loss = (student.p - 2.0) ** 2
    loss.backward()
    opt.step()
    assert _hash(behavior) == h0
    assert float(student.p) != 1.0
