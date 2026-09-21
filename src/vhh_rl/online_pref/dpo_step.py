"""Changed-position Diffusion-DPO wrapper (task book §11).

The DPO mathematics is untouched: it delegates to the validated
``compute_weighted_cf_dpo_step`` with uniform weights over the chosen support
and an explicitly supplied sigma / noise realisation.
"""
from __future__ import annotations

import torch

from .pair_builder import changed_design_positions  # noqa: F401  (re-export)
from .types import OnlinePreferencePair

SUPPORTS = ("all", "verified")


def support_positions(pair: OnlinePreferencePair, support: str) -> tuple[int, ...]:
    if support == "all":
        return tuple(int(p) for p in pair.changed_positions_all)
    if support == "verified":
        if pair.changed_positions_verified is None:
            raise ValueError(f"{pair.pair_id}: verified support requested but not built")
        return tuple(int(p) for p in pair.changed_positions_verified)
    raise ValueError(f"support must be one of {SUPPORTS}, got {support}")


def changed_position_dpo_step(policy_sm, ref_sm, feats, pair: OnlinePreferencePair,
                              network_condition_kwargs, *, support: str = "all",
                              sigma: torch.Tensor, noise: torch.Tensor,
                              beta: float = 10.0):
    from ..native_atom14.global_cf_step import compute_weighted_cf_dpo_step
    from ..native_atom14.masks import design_token_offset, residue_atom_masks

    positions = sorted(support_positions(pair, support))
    if not positions:
        raise ValueError(f"{pair.pair_id}: empty {support} support")
    design_positions = pair.meta["design_positions"]
    offset = design_token_offset(feats["token_index"], feats["design_mask"],
                                 design_positions)
    masks = residue_atom_masks(feats["atom_to_token"], feats["fake_atom_mask"],
                               feats["atom_pad_mask"],
                               [p + offset for p in positions])
    w = torch.full((len(positions),), 1.0 / len(positions),
                   device=noise.device, dtype=torch.float32)
    device = sigma.device
    return compute_weighted_cf_dpo_step(
        policy_sm, ref_sm, feats,
        pair.winner_coords.to(device), pair.loser_coords.to(device),
        masks, w, network_condition_kwargs, beta=beta,
        sigma=sigma, noise=noise)
