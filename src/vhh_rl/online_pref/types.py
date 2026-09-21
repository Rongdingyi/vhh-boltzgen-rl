"""Shared types for online preference DPO (task book §4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class OnlinePreferencePair:
    pair_id: str
    case_id: str
    round_index: int
    group_id: str

    branch_progress: float
    branch_step: int
    branch_sigma: float

    winner_index: int
    loser_index: int

    winner_reward: float
    loser_reward: float
    reward_gap: float

    winner_sequence: str
    loser_sequence: str

    # 主方法用这个
    changed_positions_all: tuple[int, ...]

    # 只作为 Phase 0 control，不进入最终方法
    changed_positions_verified: tuple[int, ...] | None

    winner_coords: torch.Tensor
    loser_coords: torch.Tensor

    conditioning: dict
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class UpdateSpec:
    update_index: int
    round_index: int
    pair_id: str

    sigma_seed: int
    noise_seed: int
    augmentation_seed: int
