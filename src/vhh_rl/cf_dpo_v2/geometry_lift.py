"""CF-DPO v2 experiment 3: legal native geometry lift (proposal §7/§12-3).

For counterfactual edges we need legal atom14 structures whose *hard decode*
equals the counterfactual sequence, with FR and background unchanged.  This
module builds them by residue-local frame transfer from the paired structure
plus hard-decode acceptance; failures are reported, never silently dropped.

Same-sequence multi-realization edges are built by decode-preserving
perturbations (rejection sampling), never by rigid moves alone.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from ..cf_opsd.target_builder import transfer_assignment
from ..native_atom14.decode import decode_atom14, fr_check, sequence_from_feat


@dataclass
class LiftAttempt:
    ok: bool
    reason: str
    coords: torch.Tensor | None = None
    sequence: str | None = None
    invalid: bool = False
    fr_mismatch: int = 0
    touched_atoms: int = 0
    moved_rms: float = 0.0


def _decode(coords: torch.Tensor, feats: dict) -> tuple[str, bool]:
    feat = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in feats.items()}
    feat["coords"] = coords.clone()
    out = decode_atom14(feat)
    return sequence_from_feat(out)[0], sequence_from_feat(out)[2]


def build_local_lift(acceptor: torch.Tensor, donor: torch.Tensor, feats: dict, site: int,
                     reference_sequence: str, fr_positions, expected_sequence: str,
                     expected_other_sequence: str, anchor_sequence: str) -> LiftAttempt:
    """Transplant donor residue `site` geometry into the acceptor structure."""
    lifted, touched = transfer_assignment(acceptor, donor, feats, [(site, site)])
    if not touched.any():
        return LiftAttempt(False, "empty_transfer")
    seq, invalid = _decode(lifted, feats)
    fr = fr_check(seq, reference_sequence, tuple(fr_positions))
    if invalid:
        return LiftAttempt(False, "decode_invalid", lifted, seq, True,
                           fr["fr_mismatch_count"], int(touched.sum()))
    if fr["fr_mismatch_count"] > 0:
        return LiftAttempt(False, "fr_changed", lifted, seq, False,
                           fr["fr_mismatch_count"], int(touched.sum()))
    moved = float(((lifted - acceptor)[touched] ** 2).sum(-1).mean().sqrt())
    if seq == expected_sequence:
        return LiftAttempt(True, "exact", lifted, seq, False, 0, int(touched.sum()), moved)
    if seq == anchor_sequence or seq == expected_other_sequence:
        return LiftAttempt(False, "decode_unchanged", lifted, seq, False, 0,
                           int(touched.sum()), moved)
    return LiftAttempt(False, "third_sequence", lifted, seq, False, 0,
                       int(touched.sum()), moved)


def build_same_sequence_realizations(coords: torch.Tensor, feats: dict,
                                     target_sequence: str, fr_positions,
                                     reference_sequence: str, *, n_realizations: int = 2,
                                     sigma: float = 0.35, max_tries: int = 60,
                                     seed: int = 0) -> list[torch.Tensor]:
    """Perturb fake atoms (local) and keep only decode-preserving structures."""
    rng = random.Random(seed)
    a2t = feats["atom_to_token"]
    if a2t.dim() == 3:
        a2t = a2t.squeeze(0)
    token_of_atom = a2t.int().argmax(-1)
    pad = feats["atom_pad_mask"].reshape(-1).bool()
    fake = feats["fake_atom_mask"].reshape(-1).bool()
    design = feats["design_mask"].reshape(-1).bool()
    movable = torch.zeros_like(pad)
    n_tokens = token_of_atom.max().item() + 1
    for t in range(int(n_tokens)):
        if t < design.numel() and bool(design[t]):
            movable |= (token_of_atom == t) & fake & pad
    out: list[torch.Tensor] = []
    for attempt in range(max_tries):
        if len(out) >= n_realizations:
            break
        noise = torch.zeros_like(coords)
        noise[movable] = torch.randn(int(movable.sum()), 3,
                                     generator=torch.Generator().manual_seed(seed * 1000 + attempt)) * sigma
        cand = coords + noise
        seq, invalid = _decode(cand, feats)
        if invalid or seq != target_sequence:
            continue
        fr = fr_check(seq, reference_sequence, tuple(fr_positions))
        if fr["fr_mismatch_count"] > 0:
            continue
        # require the perturbation to be a real difference (not a no-op)
        if float((cand - coords)[movable].abs().max()) < 1e-6:
            continue
        out.append(cand)
    return out
