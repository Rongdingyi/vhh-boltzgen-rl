"""Sequence readout helpers for branch endpoints (§18).

Only the official atom14 decoder is used; the helpers here assemble the
unbatched feat dict it expects and never reimplement geometry.
"""
from __future__ import annotations

import torch

from ..native_atom14.decode import decode_atom14, fr_check, sequence_from_feat


def squeeze_singleton_feats(feats: dict) -> dict:
    """Drop leading singleton batch dims so the unbatched decoder accepts feats."""
    out = {}
    for key, value in feats.items():
        if torch.is_tensor(value):
            while value.dim() > 1 and value.shape[0] == 1:
                value = value.squeeze(0)
        out[key] = value
    return out


def decode_coords(coords: torch.Tensor, feats: dict) -> tuple[str, bool]:
    """Decode endpoint coords to (sequence, contains_invalid)."""
    feat = {k: (v.clone() if torch.is_tensor(v) else v)
            for k, v in squeeze_singleton_feats(feats).items()}
    feat["coords"] = coords.detach().float().cpu().clone()
    out = decode_atom14(feat)
    sequence, _tokens, invalid = sequence_from_feat(out)
    return sequence, bool(invalid)


def decode_coords_with_fr(coords: torch.Tensor, feats: dict,
                          reference_sequence: str, fr_positions) -> dict:
    sequence, invalid = decode_coords(coords, feats)
    fr = fr_check(sequence, reference_sequence, tuple(fr_positions))
    return {"sequence": sequence, "contains_invalid": invalid,
            "fr_mismatch": fr["fr_mismatch_count"]}
