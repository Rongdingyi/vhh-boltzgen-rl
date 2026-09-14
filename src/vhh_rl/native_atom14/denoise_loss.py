"""Per-sample denoising loss replicating the official weighted coordinate MSE
(task book §32-34).

Replicates ``AtomDiffusion.compute_loss`` (diffusion.py:700-856) exactly for
the coordinate-MSE term, but:
  * returns shape [B] (before the final ``.mean()``) so winner/loser are not
    averaged together;
  * optionally multiplies the loss weights by a preference mask (N3/N4);
  * never adds smooth-LDDT / bond / distogram / bfactor terms.

Rigid alignment always uses the full resolved-atom set (official behaviour);
the preference mask is applied to the aligned errors afterwards, so fake atoms
cannot reduce their own loss via an independent rigid alignment (§34).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_BOLTZGEN_SRC = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/src")
if str(_BOLTZGEN_SRC) not in sys.path:
    sys.path.insert(0, str(_BOLTZGEN_SRC))

from boltzgen.data import const  # noqa: E402
from boltzgen.model.loss.diffusion import (  # noqa: E402
    weighted_rigid_align,
    weighted_rigid_centering,
)


@dataclass
class DenoiseLossOutput:
    per_sample_loss: torch.Tensor  # [B]
    per_atom_sq_error: torch.Tensor  # [B, N_atom]
    denoised_coords: torch.Tensor  # [B, N_atom, 3]
    aligned_target_coords: torch.Tensor  # [B, N_atom, 3]
    sigmas: torch.Tensor  # [B]


def _atom_align_weights(feats: dict, noised_atom_coords: torch.Tensor,
                        nucleotide_loss_weight: float, ligand_loss_weight: float) -> torch.Tensor:
    atom_type = (
        torch.bmm(
            feats["atom_to_token"].float(),
            feats["mol_type"].unsqueeze(-1).float(),
        )
        .squeeze(-1)
        .long()
    )
    weights = torch.ones_like(noised_atom_coords[:, :, 0])
    weights = weights * (
        1
        + nucleotide_loss_weight
        * (
            torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
            + torch.eq(atom_type, const.chain_type_ids["RNA"]).float()
        )
        + ligand_loss_weight
        * torch.eq(atom_type, const.chain_type_ids["NONPOLYMER"]).float()
    ).float()
    return weights


def per_sample_denoising_loss(
    structure_module,
    feats: dict,
    x0_coords: torch.Tensor,  # [1, N_atom, 3] already rigidly augmented
    noised_coords: torch.Tensor,  # [1, N_atom, 3]
    sigma: torch.Tensor,
    network_condition_kwargs: dict,
    *,
    preference_mask: torch.Tensor | None = None,  # [N_atom] or [1, N_atom] bool
    fake_atom_weight_value: float = 1.0,
    nucleotide_loss_weight: float = 5.0,
    ligand_loss_weight: float = 10.0,
    require_grad: bool = True,
) -> DenoiseLossOutput:
    if require_grad:
        denoised, _net = structure_module.preconditioned_network_forward(
            noised_coords,
            sigma.reshape(-1),
            training=True,
            network_condition_kwargs=network_condition_kwargs,
        )
    else:
        with torch.no_grad():
            denoised, _net = structure_module.preconditioned_network_forward(
                noised_coords,
                sigma.reshape(-1),
                training=True,
                network_condition_kwargs=network_condition_kwargs,
            )

    denoised = denoised.float()
    noised = noised_coords.float()
    sigmas = sigma.reshape(-1).float()

    resolved = feats["atom_resolved_mask"].float()
    if resolved.dim() == 1:
        resolved = resolved.unsqueeze(0)
    fake_atom_mask = feats["fake_atom_mask"].float()
    if fake_atom_mask.dim() == 1:
        fake_atom_mask = fake_atom_mask.unsqueeze(0)
    fake_atom_weight = (1 - fake_atom_mask) + fake_atom_mask * fake_atom_weight_value

    align_weights = _atom_align_weights(
        feats, noised, nucleotide_loss_weight, ligand_loss_weight
    )

    target = x0_coords.float()
    if structure_module.mse_rotational_alignment:
        aligned_target = weighted_rigid_align(
            target.detach(), denoised.detach(), align_weights.detach(),
            mask=resolved.detach(),
        )
    else:
        aligned_target = weighted_rigid_centering(
            target, denoised, align_weights, mask=resolved,
        )

    sq_error = ((denoised - aligned_target.detach()) ** 2).sum(dim=-1)  # [B, N_atom]
    weights = align_weights * fake_atom_weight * resolved
    if preference_mask is not None:
        pm = preference_mask.float()
        if pm.dim() == 1:
            pm = pm.unsqueeze(0)
        weights = weights * pm
    per_sample = sq_error.mul(weights).sum(dim=-1) / (
        3.0 * weights.sum(dim=-1) + 1e-5
    )
    loss_weights = structure_module.loss_weight(sigmas)
    per_sample = per_sample * loss_weights
    return DenoiseLossOutput(
        per_sample_loss=per_sample,
        per_atom_sq_error=sq_error,
        denoised_coords=denoised,
        aligned_target_coords=aligned_target.detach(),
        sigmas=sigmas,
    )


def per_residue_denoising_loss(
    structure_module,
    feats: dict,
    x0_coords: torch.Tensor,  # [1, N_atom, 3] already rigidly augmented
    noised_coords: torch.Tensor,  # [1, N_atom, 3]
    sigma: torch.Tensor,
    network_condition_kwargs: dict,
    residue_masks: torch.Tensor,  # [R, N_atom] bool
    *,
    fake_atom_weight_value: float = 1.0,
    nucleotide_loss_weight: float = 5.0,
    ligand_loss_weight: float = 10.0,
    require_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-residue coordinate losses (task book §35).

    Global rigid alignment uses the full resolved-atom set exactly like the
    official loss; only the error aggregation is per residue.  Returns
    (losses [R], denoised_coords [1, N_atom, 3]); the caller applies the sigma
    loss weight after its own (credit-weighted) aggregation.
    """
    if require_grad:
        denoised, _net = structure_module.preconditioned_network_forward(
            noised_coords, sigma.reshape(-1), training=True,
            network_condition_kwargs=network_condition_kwargs,
        )
    else:
        with torch.no_grad():
            denoised, _net = structure_module.preconditioned_network_forward(
                noised_coords, sigma.reshape(-1), training=True,
                network_condition_kwargs=network_condition_kwargs,
            )
    denoised = denoised.float()
    noised = noised_coords.float()
    resolved = feats["atom_resolved_mask"].float()
    if resolved.dim() == 1:
        resolved = resolved.unsqueeze(0)
    fake_atom_mask = feats["fake_atom_mask"].float()
    if fake_atom_mask.dim() == 1:
        fake_atom_mask = fake_atom_mask.unsqueeze(0)
    fake_atom_weight = (1 - fake_atom_mask) + fake_atom_mask * fake_atom_weight_value
    align_weights = _atom_align_weights(
        feats, noised, nucleotide_loss_weight, ligand_loss_weight
    )
    target = x0_coords.float()
    if structure_module.mse_rotational_alignment:
        aligned_target = weighted_rigid_align(
            target.detach(), denoised.detach(), align_weights.detach(),
            mask=resolved.detach(),
        )
    else:
        aligned_target = weighted_rigid_centering(
            target, denoised, align_weights, mask=resolved,
        )
    sq_error = ((denoised - aligned_target.detach()) ** 2).sum(dim=-1)  # [1, N]
    weights = align_weights * fake_atom_weight * resolved  # [1, N]
    masks = residue_masks.float()
    if masks.dim() == 1:
        masks = masks.unsqueeze(0)
    num = (sq_error.unsqueeze(0) * weights.unsqueeze(0) * masks).sum(dim=-1)  # [R]
    den = 3.0 * (weights.unsqueeze(0) * masks).sum(dim=-1) + 1e-8
    return num / den, denoised


def sample_denoising_loss(
    structure_module,
    feats: dict,
    coords: torch.Tensor,  # [N_atom or 1, N_atom, 3] unaugmented X0
    network_condition_kwargs: dict,
    *,
    preference_mask: torch.Tensor | None = None,
    **kwargs,
) -> DenoiseLossOutput:
    """Convenience: official-style augmentation + noise + per-sample loss.

    Used for N1 RWR (no pairing needed).
    """
    from .paired_noise import paired_noising

    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    atom_mask = feats["atom_pad_mask"]
    if atom_mask.dim() == 1:
        atom_mask = atom_mask.unsqueeze(0)
    sigma = structure_module.noise_distribution(coords.shape[0])
    noise = torch.randn_like(coords)
    # centered(+rotated) copy of X0 is what the official forward uses as target
    from boltzgen.model.modules.utils import center_random_augmentation

    x0_aug = center_random_augmentation(
        coords, atom_mask.float(),
        augmentation=structure_module.coordinate_augmentation,
    )
    noised = x0_aug + sigma.reshape(-1, 1, 1) * noise
    return per_sample_denoising_loss(
        structure_module, feats, x0_aug, noised, sigma,
        network_condition_kwargs,
        preference_mask=preference_mask,
        **kwargs,
    )
