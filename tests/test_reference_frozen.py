"""Reference must stay frozen (task book §31/§67) — pure torch check."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


def test_reference_copy_not_updated():
    policy = nn.Linear(4, 4)
    reference = copy.deepcopy(policy)
    for p in reference.parameters():
        p.requires_grad_(False)
    ref_before = [p.detach().clone() for p in reference.parameters()]
    opt = torch.optim.SGD(policy.parameters(), lr=0.5)
    x = torch.randn(3, 4)
    loss = policy(x).pow(2).mean()
    loss.backward()
    opt.step()
    for before, after in zip(ref_before, reference.parameters()):
        assert torch.equal(before, after)
    # policy actually moved
    assert any(not torch.equal(b, p) for b, p in zip(ref_before, policy.parameters()))
