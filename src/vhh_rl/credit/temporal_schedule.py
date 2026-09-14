"""Temporal credit schedule (task book §29).

g(sigma) = sigmoid((log sigma_seq - log sigma) / tau); tau fixed at 0.5,
no sweep.  g_hard is the ablation version.
"""
from __future__ import annotations

import math


def g_smooth(sigma: float, sigma_seq: float, tau: float = 0.5) -> float:
    if sigma <= 0 or sigma_seq <= 0:
        return 0.0
    x = (math.log(sigma_seq) - math.log(sigma)) / tau
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def g_hard(sigma: float, sigma_seq: float) -> float:
    return 1.0 if sigma <= sigma_seq else 0.0
