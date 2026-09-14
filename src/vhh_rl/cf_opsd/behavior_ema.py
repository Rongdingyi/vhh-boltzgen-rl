"""Behavior policy EMA refresh (task book §57)."""
from __future__ import annotations

import copy

import torch


def ema_update(behavior_sm, student_sm, decay: float = 0.99) -> None:
    """behavior <- decay * behavior + (1 - decay) * student (in place)."""
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must be in (0,1)")
    b_params = dict(behavior_sm.named_parameters())
    with torch.no_grad():
        for name, param in student_sm.named_parameters():
            if name in b_params:
                b_params[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)


def copy_student_to_behavior(behavior_sm, student_sm) -> None:
    behavior_sm.load_state_dict(copy.deepcopy(student_sm.state_dict()), strict=False)
