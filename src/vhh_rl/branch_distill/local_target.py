"""Changed-residue local geometry target (task book §3/§19/§29/§30).

The ONLY target construction allowed here is residue-local frame transfer of
the reward-selected real teacher endpoint onto the peer query anchor; no
radius, no credit, no interpolation.
"""
from __future__ import annotations

import torch

from ..cf_opsd.local_frame import transfer_winner_to_anchor
from .decode import decode_coords_with_fr


def changed_design_positions(teacher_sequence: str, peer_sequence: str,
                             design_positions) -> tuple[int, ...]:
    """Design positions where the two endpoint sequences differ (§19)."""
    return tuple(p for p in sorted(design_positions)
                 if teacher_sequence[p] != peer_sequence[p])


def build_target(peer_anchor: torch.Tensor, teacher_endpoint: torch.Tensor,
                 feats: dict, changed_positions) -> tuple[torch.Tensor, torch.Tensor]:
    """transfer_winner_to_anchor(peer_anchor, teacher_endpoint, feats, M)."""
    return transfer_winner_to_anchor(peer_anchor, teacher_endpoint, feats,
                                     list(changed_positions))


def target_mask(touched: torch.Tensor, feats: dict) -> torch.Tensor:
    """§36: touched_mask & fake_atom_mask & atom_pad_mask."""
    pad = feats["atom_pad_mask"].reshape(-1).bool()
    fake = feats["fake_atom_mask"].reshape(-1).bool()
    return touched.reshape(-1).bool() & fake & pad


def carrier_audit(peer_endpoint: torch.Tensor, teacher_endpoint: torch.Tensor,
                  feats: dict, changed_positions,
                  reference_sequence: str, fr_positions) -> dict:
    """§30 carrier decode: transfer onto a legal endpoint must decode teacher AA.

    The peer anchor may itself be invalid early in the trajectory, so the
    carrier check is run on the peer *endpoint* geometry.
    """
    carrier, _touched = transfer_winner_to_anchor(
        peer_endpoint, teacher_endpoint, feats, list(changed_positions))
    audit = decode_coords_with_fr(carrier, feats, reference_sequence, fr_positions)
    return {"sequence": audit["sequence"],
            "contains_invalid": audit["contains_invalid"],
            "fr_mismatch": audit["fr_mismatch"],
            "carrier_coords": carrier}


def carrier_positions_match(carrier_sequence: str, teacher_sequence: str,
                            changed_positions) -> dict:
    """Per-position AA equality of carrier decode vs teacher endpoint sequence."""
    return {p: bool(carrier_sequence[p] == teacher_sequence[p])
            for p in changed_positions}
