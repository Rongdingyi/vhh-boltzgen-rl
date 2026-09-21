"""Pure sequence-difference preference pairs (task book §6).

The main path never touches geometry: a pair is built from valid sibling
endpoints, reward ranking and the design-position sequence difference only.
"""
from __future__ import annotations

import statistics as st

import torch

from .types import OnlinePreferencePair


def changed_design_positions(winner_sequence: str, loser_sequence: str,
                             design_positions) -> tuple[int, ...]:
    return tuple(int(p) for p in sorted(design_positions)
                 if winner_sequence[p] != loser_sequence[p])


def valid_siblings(siblings):
    """Valid = decodable, FR-clean, scored."""
    return [s for s in siblings
            if not s.contains_invalid and s.fr_mismatch == 0 and s.reward is not None]


def ranked_valid(siblings):
    return sorted(valid_siblings(siblings), key=lambda s: s.reward, reverse=True)


def _peer(ranked, peer_type: str):
    if peer_type == "worst":
        return ranked[-1]
    if peer_type == "median":
        rewards = sorted(s.reward for s in ranked)
        target = st.median(rewards)
        return min(ranked, key=lambda s: abs(s.reward - target))
    raise ValueError(f"unknown peer_type {peer_type}")


def build_all_changed_pairs(group, design_positions, *, round_index: int = 0,
                            peer_types=("worst", "median"),
                            min_reward_gap: float = 0.30,
                            ) -> list[OnlinePreferencePair]:
    """best-vs-worst / best-vs-median over the *all-changed* support (§6.2)."""
    ranked = ranked_valid(group.siblings)
    if len(ranked) < 2:
        return []
    winner = ranked[0]
    pairs = []
    design_positions = tuple(sorted(int(p) for p in design_positions))
    for peer_type in peer_types:
        loser = _peer(ranked, peer_type)
        if loser.branch_index == winner.branch_index:
            continue
        changed = changed_design_positions(winner.endpoint_sequence,
                                           loser.endpoint_sequence,
                                           design_positions)
        gap = float(winner.reward - loser.reward)
        if gap < min_reward_gap or len(changed) < 1:
            continue
        pair_id = f"{group.case_id}:{group.group_seed}:{peer_type}"
        pairs.append(OnlinePreferencePair(
            pair_id=pair_id,
            case_id=group.case_id,
            round_index=int(round_index),
            group_id=f"{group.case_id}:{group.group_seed}",
            branch_progress=float(group.progress),
            branch_step=int(group.start_step),
            branch_sigma=float(group.sigma),
            winner_index=int(winner.branch_index),
            loser_index=int(loser.branch_index),
            winner_reward=float(winner.reward),
            loser_reward=float(loser.reward),
            reward_gap=gap,
            winner_sequence=winner.endpoint_sequence,
            loser_sequence=loser.endpoint_sequence,
            changed_positions_all=changed,
            changed_positions_verified=None,
            winner_coords=winner.endpoint_coords.detach().float().cpu().clone(),
            loser_coords=loser.endpoint_coords.detach().float().cpu().clone(),
            conditioning=group.conditioning,
            meta={"peer_type": peer_type,
                  "design_positions": design_positions,
                  "branch_progress": float(group.progress),
                  "branch_step": int(group.start_step),
                  "branch_sigma": float(group.sigma),
                  "n_valid_siblings": len(ranked)},
        ))
    return pairs


def build_pairs_for_groups(groups, design_positions_by_case, *, round_index: int = 0,
                           peer_types=("worst", "median"),
                           min_reward_gap: float = 0.30):
    out = []
    for group in groups:
        design = design_positions_by_case[group.case_id]
        out.extend(build_all_changed_pairs(
            group, design, round_index=round_index, peer_types=peer_types,
            min_reward_gap=min_reward_gap))
    return out
