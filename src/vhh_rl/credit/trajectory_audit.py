"""Phase B1 trajectory audit metrics (task book §20-§28).

Pure torch/CPU geometry helpers: FR-backbone Kabsch alignment, per-step
sequence convergence (identity / stable fraction), sigma threshold search.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch


def kabsch_rmsd(pred: torch.Tensor, ref: torch.Tensor,
                align_mask: torch.Tensor, eval_mask: torch.Tensor) -> float:
    """RMSD of pred vs ref over eval_mask after rigid-aligning on align_mask."""
    a = pred[align_mask].float()
    b = ref[align_mask].float()
    if a.shape[0] < 3 or eval_mask.sum() == 0:
        return float("nan")
    a_c = a - a.mean(0, keepdim=True)
    b_c = b - b.mean(0, keepdim=True)
    h = a_c.T @ b_c
    u, _s, vt = torch.linalg.svd(h)
    d = torch.sign(torch.det(vt.T @ u.T))
    r = vt.T @ torch.diag(torch.tensor([1.0, 1.0, d], dtype=a.dtype)) @ u.T
    aligned = (pred.float() - a.mean(0, keepdim=True)) @ r.T + b.mean(0, keepdim=True)
    diff = aligned[eval_mask] - ref.float()[eval_mask]
    return float((diff ** 2).sum(-1).mean().sqrt())


def cdr_identity(seq: str, final: str, design_positions: Sequence[int]) -> float:
    if not design_positions:
        return float("nan")
    return sum(1 for i in design_positions if seq[i] == final[i]) / len(design_positions)


def identity_all(seq: str, final: str) -> float:
    return sum(1 for a, b in zip(seq, final) if a == b) / max(len(final), 1)


def stable_fraction(seqs_from_t: Sequence[str], final: str,
                    design_positions: Sequence[int]) -> float:
    """Fraction of design residues whose decoded AA from t to the end never changes."""
    if not seqs_from_t or not design_positions:
        return float("nan")
    n_stable = 0
    for pos in design_positions:
        if all(s[pos] == final[pos] for s in seqs_from_t):
            n_stable += 1
    return n_stable / len(design_positions)


def find_sigma_threshold(
    sigmas: Sequence[float],
    median_identity: Sequence[float],
    valid_rate: Sequence[float],
    min_identity: float = 0.80,
    min_valid: float = 0.90,
) -> float | None:
    """Largest sigma whose entire trailing sequence (smaller sigma) satisfies
    median identity >= min_identity AND valid rate >= min_valid."""
    n = len(sigmas)
    threshold = None
    for i in range(n - 1, -1, -1):
        if median_identity[i] >= min_identity and valid_rate[i] >= min_valid:
            threshold = sigmas[i]
        else:
            break
    return threshold


def normalized_convergence(rmsds: Sequence[float]) -> list[float]:
    """c(t) = 1 - rmsd(t) / max(rmsd), robust to non-monotone curves."""
    finite = [r for r in rmsds if not math.isnan(r)]
    if not finite:
        return [float("nan")] * len(rmsds)
    scale = max(finite) if max(finite) > 0 else 1.0
    return [float("nan") if math.isnan(r) else max(0.0, 1.0 - r / scale) for r in rmsds]
