"""Phase B: query selection on the denoising trajectory (task book §13-§17)."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..native_atom14.decode import decode_atom14, sequence_from_feat
from .types import OPSDTrajectory

CANDIDATE_PROGRESS = (0.60, 0.70, 0.80, 0.90)


def progress_to_step(progress: float, n_steps: int) -> int:
    """0% = start of denoising (first network call), 100% = final call."""
    idx = int(round(progress * n_steps)) - 1
    return max(0, min(n_steps - 1, idx))


def decode_anchor(traj: OPSDTrajectory, step: int) -> tuple[str, bool]:
    feats = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in traj.feats_common.items()}
    feats["coords"] = traj.anchors[step].clone()
    out = decode_atom14(feats)
    seq, _tokens, invalid = sequence_from_feat(out)
    return seq, invalid


def cdr_identity(seq: str, ref: str, positions) -> float:
    return sum(1 for p in positions if seq[p] == ref[p]) / max(1, len(positions))


@dataclass
class QueryCandidate:
    progress: float
    step: int
    valid_rate: float
    median_cdr_identity: float
    median_full_identity: float


def evaluate_candidates(trajectories: list[OPSDTrajectory], design_positions,
                        progresses=CANDIDATE_PROGRESS) -> list[QueryCandidate]:
    n_steps = len(trajectories[0].sigmas)
    out = []
    for p in progresses:
        step = progress_to_step(p, n_steps)
        valid, cdr, full = [], [], []
        for traj in trajectories:
            seq, invalid = decode_anchor(traj, step)
            valid.append(0.0 if invalid else 1.0)
            cdr.append(cdr_identity(seq, traj.endpoint_sequence, design_positions))
            full.append(cdr_identity(seq, traj.endpoint_sequence,
                                     range(len(traj.endpoint_sequence))))
        out.append(QueryCandidate(
            progress=p, step=step,
            valid_rate=sum(valid) / len(valid),
            median_cdr_identity=sorted(cdr)[len(cdr) // 2],
            median_full_identity=sorted(full)[len(full) // 2],
        ))
    return out


def select_query(candidates: list[QueryCandidate], min_valid: float = 0.90,
                 min_cdr: float = 0.80) -> tuple[QueryCandidate, str]:
    """Earliest progress meeting both thresholds; else 90% with weak flag."""
    for c in candidates:  # candidates are ordered 60 -> 90
        if c.valid_rate >= min_valid and c.median_cdr_identity >= min_cdr:
            return c, "PASS"
    fallback = candidates[-1]
    status = "QUERY_ANCHOR_WEAK" if fallback.valid_rate >= 0.80 else "NO_GO"
    return fallback, status
