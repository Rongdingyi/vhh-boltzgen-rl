"""Phase B: bounded positive atom14 target construction (task book §25-§28).

Y+ = A + s * D_weighted,  D = T(winner -> anchor) - A
weighted with sqrt(u_i / u_bar) (default) and s = min(1, rho / rms(D)).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

from .local_frame import transfer_winner_to_anchor


def credit_weights(credits: dict[int, float], n_diff: int, eta: float = 0.75) -> dict[int, float]:
    floor = (1.0 - eta) / max(1, n_diff)
    total = sum(c for c in credits.values() if c > 0)
    if total <= 0:
        return {p: 1.0 / max(1, n_diff) for p in credits}
    return {p: floor + eta * (max(c, 0.0) / total) for p, c in credits.items()}


def transfer_assignment(anchor: torch.Tensor, winner: torch.Tensor, feats: dict,
                        assignments: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized transfer: anchor position <- winner position geometry."""
    a2t = feats["atom_to_token"]
    if a2t.dim() == 3 and a2t.shape[0] == 1:
        a2t = a2t.squeeze(0)
    token_of_atom = a2t.int().argmax(-1)
    fake = feats["fake_atom_mask"].reshape(-1).bool()
    pad = feats["atom_pad_mask"].reshape(-1).bool()

    from .local_frame import backbone_indices, frame_from_backbone, to_local, to_world

    out = anchor.clone().float()
    touched = torch.zeros(anchor.shape[0], dtype=torch.bool)
    anchor_backbones = backbone_indices(feats, [a for a, _w in assignments])
    winner_backbones = backbone_indices(feats, [w for _a, w in assignments])
    for (a_pos, w_pos), (an, aca, ac), (wn, wca, wc) in zip(
            assignments, anchor_backbones, winner_backbones):
        F_a, o_a = frame_from_backbone(anchor, an, aca, ac)
        F_w, o_w = frame_from_backbone(winner, wn, wca, wc)
        atoms = torch.where((token_of_atom == int(a_pos)) & fake & pad)[0]
        if atoms.numel() == 0:
            continue
        local = to_local(winner[atoms], F_w, o_w)
        out[atoms] = to_world(local, F_a, o_a)
        touched[atoms] = True
    return out, touched


@dataclass
class BuiltTarget:
    target_coords: torch.Tensor
    touched: torch.Tensor
    radius: float
    control: str
    actual_rms: float
    saturation: float
    weights: dict[int, float] = field(default_factory=dict)


def build_target(anchor: torch.Tensor, winner: torch.Tensor, feats: dict,
                 differing_positions: list[int], credits: dict[int, float],
                 radius: float, *, eta: float = 0.75, credit_scale: str = "sqrt",
                 control: str = "cf", rng: random.Random | None = None) -> BuiltTarget:
    rng = rng or random.Random(0)
    n_diff = len(differing_positions)
    credited = [p for p in differing_positions if credits.get(p, 0.0) > 0]
    if control == "noop":
        assignments = []
        weights = {}
    elif control == "random":
        k = max(1, len(credited))
        picks = rng.sample(differing_positions, min(k, len(differing_positions)))
        assignments = [(p, p) for p in picks]
        weights = {p: 1.0 / len(picks) for p in picks}
    elif control == "shuffle":
        _w = credit_weights({p: credits.get(p, 0.0) for p in credited}, n_diff, eta)
        sources = [p for p in credited]
        rng.shuffle(sources)
        assignments = list(zip(credited, sources))
        weights = _w
    else:  # cf
        weights = credit_weights({p: credits.get(p, 0.0) for p in credited}, n_diff, eta)
        assignments = [(p, p) for p in credited]

    if not assignments:
        return BuiltTarget(anchor.clone().float(),
                           torch.zeros(anchor.shape[0], dtype=torch.bool),
                           radius, control, 0.0, 0.0, weights)

    transferred, touched = transfer_assignment(anchor, winner, feats, assignments)
    D = transferred - anchor
    D[~touched] = 0.0

    if credit_scale == "sqrt" and weights:
        u_bar = sum(weights.values()) / len(weights)
        scale = torch.ones(anchor.shape[0])
        token_of_atom = feats["atom_to_token"]
        if token_of_atom.dim() == 3:
            token_of_atom = token_of_atom.squeeze(0)
        token_of_atom = token_of_atom.int().argmax(-1)
        for p, u in weights.items():
            factor = (u / u_bar) ** 0.5
            atoms = torch.where((token_of_atom == int(p)) & touched)[0]
            if atoms.numel():
                scale[atoms] = factor
        D = D * scale[:, None]
    elif credit_scale == "linear" and weights:
        # linear variant scales by u_i / u_bar (documented, not default)
        u_bar = sum(weights.values()) / len(weights)
        token_of_atom = feats["atom_to_token"]
        if token_of_atom.dim() == 3:
            token_of_atom = token_of_atom.squeeze(0)
        token_of_atom = token_of_atom.int().argmax(-1)
        for p, u in weights.items():
            atoms = torch.where((token_of_atom == int(p)) & touched)[0]
            if atoms.numel():
                D[atoms] = D[atoms] * (u / u_bar)

    rms = float((D[touched] ** 2).sum(-1).mean().sqrt()) if touched.any() else 0.0
    s = min(1.0, radius / (rms + 1e-8))
    Y = anchor + s * D
    actual = rms * s
    return BuiltTarget(Y, touched, radius, control, actual,
                       1.0 if s >= 1.0 - 1e-9 else 0.0, weights)
