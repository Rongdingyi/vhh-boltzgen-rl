"""Group advantages computed WITHIN one case (task book §27/§28)."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import torch


def group_advantages(
    rewards: dict[str, Sequence[float]],
    *,
    method: str,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, int]:
    """Per-case normalization. Returns flat advantages in the dict's iteration
    order plus the number of zero-variance groups."""
    advantages: list[torch.Tensor] = []
    zero_var = 0
    for _, values in rewards.items():
        t = torch.tensor([float(v) for v in values], dtype=torch.float32)
        if method == "group_z":
            mu, std = t.mean(), t.std(unbiased=False)
            if float(std) < eps or math.isnan(float(std)):
                zero_var += 1
                advantages.append(torch.zeros_like(t))
            else:
                advantages.append((t - mu) / (std + eps))
        elif method == "rank":
            # average ranks for ties, mapped to [-1, 1], group mean ~ 0
            n = t.numel()
            order = t.argsort().tolist()
            ranks = torch.empty(n, dtype=torch.float32)
            i = 0
            while i < n:
                j = i
                while j + 1 < n and t[order[j + 1]] == t[order[i]]:
                    j += 1
                avg = (i + j) / 2.0  # zero-based average rank of the tie block
                for k in range(i, j + 1):
                    ranks[order[k]] = avg
                i = j + 1
            if n > 1:
                ranks = (ranks / (n - 1)) * 2.0 - 1.0
            else:
                ranks = torch.zeros_like(ranks)
            advantages.append(ranks)
        else:
            raise ValueError(f"unknown advantage type {method!r}")
    if not advantages:
        return torch.empty(0), 0
    return torch.cat(advantages), zero_var
