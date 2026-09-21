"""Synthetic fixtures for online_pref tests."""
from __future__ import annotations

import torch

from vhh_rl.branch_distill.types import BranchGroup, SiblingEndpoint


def make_sibling(index: int, reward, sequence: str, invalid: bool = False, fr: int = 0):
    coords = torch.full((5, 3), float(index))
    return SiblingEndpoint(branch_index=index, endpoint_coords=coords,
                           endpoint_sequence=sequence, reward=reward,
                           contains_invalid=invalid, fr_mismatch=fr,
                           query_coords=coords.clone(), anchor_coords=coords.clone())


def make_group(case_id="case_a", rewards=(0.0, 1.0, 2.0), sequences=("AAAA", "AAAB", "AABB"),
               sigma=5.0, progress=0.60, step=29, seed=7):
    siblings = [make_sibling(i, r, s) for i, (r, s) in enumerate(zip(rewards, sequences))]
    return BranchGroup(case_id=case_id, source_sample_index=0, progress=progress,
                       start_step=step, sigma=sigma, group_seed=seed,
                       pre_state=torch.zeros(5, 3), full_query_batch=torch.zeros(3, 5, 3),
                       full_anchor_batch=torch.zeros(3, 5, 3), siblings=siblings,
                       conditioning={"feats": {}, "s_inputs": torch.zeros(1),
                                     "s_trunk": torch.zeros(1),
                                     "diffusion_conditioning": {}},
                       meta={"design_positions": (1, 2, 3)})
