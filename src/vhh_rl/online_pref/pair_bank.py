"""Per-round pair banks (task book §8): full sibling groups + pairs."""
from __future__ import annotations

from pathlib import Path

import torch

from .types import OnlinePreferencePair


def save_pair_bank(path, *, round_index: int, behavior_checkpoint_sha256: str,
                   branch_progress: float, groups, pairs, reward_queries: int):
    payload = {
        "round": int(round_index),
        "behavior_checkpoint_sha256": behavior_checkpoint_sha256,
        "branch_progress": float(branch_progress),
        "groups": [g.__dict__ for g in groups],
        "pairs": [p.__dict__ for p in pairs],
        "reward_queries": int(reward_queries),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def load_pair_bank(path) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    payload["pairs"] = [OnlinePreferencePair(**row) for row in payload["pairs"]]
    return payload


def pair_bank_path(root, seed: int, round_index: int) -> Path:
    return Path(root) / f"seed_{seed}" / f"pair_bank_r{round_index}.pt"


def schedule_path(root, seed: int, round_index: int) -> Path:
    return Path(root) / f"seed_{seed}" / f"update_schedule_r{round_index}.json"
