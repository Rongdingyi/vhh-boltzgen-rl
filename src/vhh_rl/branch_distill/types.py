"""Shared types for on-policy sibling branching + local distillation (§9)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class SiblingEndpoint:
    branch_index: int
    endpoint_coords: torch.Tensor
    endpoint_sequence: str
    reward: float | None
    contains_invalid: bool
    fr_mismatch: int
    query_coords: torch.Tensor
    anchor_coords: torch.Tensor
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BranchGroup:
    case_id: str
    source_sample_index: int
    progress: float
    start_step: int
    sigma: float
    group_seed: int
    pre_state: torch.Tensor
    full_query_batch: torch.Tensor
    full_anchor_batch: torch.Tensor
    siblings: list[SiblingEndpoint]
    conditioning: dict
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistillRecord:
    record_id: str
    case_id: str
    progress: float
    start_step: int
    group_seed: int
    branch_count: int
    teacher_index: int
    peer_index: int
    teacher_reward: float
    peer_reward: float
    reward_gap: float
    teacher_sequence: str
    peer_sequence: str
    changed_positions: tuple[int, ...]
    full_query_batch: torch.Tensor
    full_anchor_batch: torch.Tensor
    peer_anchor: torch.Tensor
    target_coords: torch.Tensor
    touched_mask: torch.Tensor
    pre_state: torch.Tensor
    conditioning: dict
    meta: dict[str, Any] = field(default_factory=dict)
