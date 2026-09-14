"""CF-OPSD shared types (task book §9/§48)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class OPSDTrajectory:
    case_id: str
    sample_index: int
    seed: int

    endpoint_coords: torch.Tensor          # [N_atom, 3] fp32 cpu
    endpoint_sequence: str
    endpoint_reward: float | None

    sigmas: list[float]                    # per network call (t_hat)
    coords_traj: list[torch.Tensor]        # [N,3] per sampler state (T+1)
    query_states: list[torch.Tensor]       # exact noised input per call (T)
    anchors: list[torch.Tensor]            # exact clean prediction per call (T)

    feats_common: dict[str, torch.Tensor] = field(default_factory=dict)
    contains_invalid: bool = False
    fr_mismatch: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetRecord:
    case_id: str
    trajectory_seed: int
    query_index: int
    query_progress: float
    sigma: float
    query_coords: torch.Tensor
    anchor_coords: torch.Tensor
    winner_coords: torch.Tensor
    loser_coords: torch.Tensor
    winner_sequence: str
    loser_sequence: str
    credits: dict[int, float]              # position -> c_cons
    target_coords: torch.Tensor
    target_sequence: str | None
    target_reward: float | None
    radius: float
    control: str = "cf"
    meta: dict[str, Any] = field(default_factory=dict)
