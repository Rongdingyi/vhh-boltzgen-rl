"""Paired noising for Diffusion-DPO (task book §28-30, §49).

Hard requirement: winner and loser share the same sigma, the same Gaussian
noise, and the same random rigid augmentation (rotation + translation).

The shared rigid augmentation is delegated to the official
``center_random_augmentation`` with ``return_second_coords=True`` /
``second_coords=<loser>`` (utils.py:68): one rotation + one translation are
drawn and applied to both coordinate sets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_BOLTZGEN_SRC = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/src")
if str(_BOLTZGEN_SRC) not in sys.path:
    sys.path.insert(0, str(_BOLTZGEN_SRC))

from boltzgen.model.modules.utils import center_random_augmentation  # noqa: E402


def paired_noising(
    winner_coords: torch.Tensor,  # [N_atom, 3]
    loser_coords: torch.Tensor,  # [N_atom, 3]
    atom_mask: torch.Tensor,  # [N_atom] bool
    sigma: torch.Tensor,  # scalar tensor
    *,
    augmentation: bool = True,
    noise: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    """Return synchronized noised states for a winner/loser pair.

    X0 -> same rigid augmentation -> + same sigma * same noise.
    """
    if winner_coords.shape != loser_coords.shape:
        raise ValueError("winner/loser must share the atom layout")
    device = winner_coords.device
    w = winner_coords[None]  # [1, N, 3]
    l = loser_coords[None]
    mask = atom_mask[None].to(device).float()
    if noise is not None and noise.device != device:
        noise = noise.to(device)
    w_aug, l_aug = center_random_augmentation(
        w, mask,
        augmentation=augmentation,
        return_second_coords=True,
        second_coords=l,
    )
    if noise is None:
        noise = torch.randn_like(w_aug)
    if noise.shape != w_aug.shape:
        raise ValueError(f"noise shape {tuple(noise.shape)} != coords {tuple(w_aug.shape)}")
    sig = sigma.reshape(1, 1, 1)
    xt_w = w_aug + sig * noise
    xt_l = l_aug + sig * noise
    return {
        "x0_w_aug": w_aug,
        "x0_l_aug": l_aug,
        "x_t_w": xt_w,
        "x_t_l": xt_l,
        "sigma": sigma,
        "noise": noise,
    }


def sample_sigma(structure_module, batch_size: int = 1) -> torch.Tensor:
    """Official noise_distribution (diffusion.py:636)."""
    return structure_module.noise_distribution(batch_size)
