"""Resume the official BoltzGen sampler from an exact captured state (§12-§17).

The loop below is a line-by-line replica of
``AtomDiffusion.sample`` at BoltzGen commit
a3149cf18eeb58648d1abbb27539bd73f746cdda, restricted to the tail
``[start_step, num_sampling_steps)``.  No sampler equation is modified:
schedule, gamma, step/noise scales, centering, coordinate augmentation,
noise draw, preconditioned forward, alignment reverse diff and the Euler
update all come from the official code.

Heavy BoltzGen imports are lazy so the module stays importable on CPU-only
CI (tests monkeypatch the ``center`` / ``compute_random_augmentation`` /
``weighted_rigid_align`` names).
"""
from __future__ import annotations

from math import sqrt

import torch

DEVICE = "cuda"


def _boltz_helpers():
    from boltzgen.model.loss.diffusion import weighted_rigid_align
    from boltzgen.model.modules.utils import center, compute_random_augmentation
    return center, compute_random_augmentation, weighted_rigid_align


def _default(value, fallback):
    if value is None and fallback is None:
        raise ValueError("no step/noise scale available: pass the scales captured "
                         "by collect_case_rollouts (info['sampling_scales'])")
    return value if value is not None else fallback


def schedule_arrays(diffusion, num_sampling_steps: int, *, device=None,
                    step_scale=None, noise_scale=None):
    """Official sigmas / gammas / step scales / noise scales (verbatim)."""
    device = device or DEVICE
    if diffusion.training and diffusion.step_scale_random is not None:
        import numpy as np
        step_scales = (np.random.choice(diffusion.step_scale_random)
                       * torch.ones(num_sampling_steps, device=device,
                                    dtype=torch.float32))
    elif diffusion.step_scale_function == "beta":
        step_scales = diffusion.beta_step_scale_schedule(num_sampling_steps)
    else:
        step_scales = _default(step_scale, diffusion.step_scale) * torch.ones(
            num_sampling_steps, device=device, dtype=torch.float32)
    if diffusion.noise_scale_function == "constant":
        noise_scales = _default(noise_scale, diffusion.noise_scale) * torch.ones(
            num_sampling_steps, device=device, dtype=torch.float32)
    elif diffusion.noise_scale_function == "beta":
        noise_scales = diffusion.beta_noise_scale_schedule(num_sampling_steps)
    else:
        raise ValueError(f"Invalid noise scale schedule: {diffusion.noise_scale_function}")
    if diffusion.sampling_schedule == "af3":
        sigmas = diffusion.sample_schedule_af3(num_sampling_steps)
    elif diffusion.sampling_schedule == "dilated":
        sigmas = diffusion.sample_schedule_dilated(num_sampling_steps)
    else:
        raise ValueError(f"unknown sampling schedule {diffusion.sampling_schedule}")
    gammas = torch.where(sigmas > diffusion.gamma_min, diffusion.gamma_0, 0.0)
    return sigmas, gammas, step_scales, noise_scales


@torch.no_grad()
def continue_from_state(
    diffusion,
    *,
    pre_state: torch.Tensor,
    start_step: int,
    num_sampling_steps: int,
    multiplicity: int,
    atom_mask: torch.Tensor,
    network_condition_kwargs: dict,
    seed: int,
    step_scale=None,
    noise_scale=None,
    assert_synchronized: bool = True,
    device=None,
) -> dict:
    """Sample K siblings from one exact state, official equations only."""
    if not 0 <= start_step < num_sampling_steps:
        raise ValueError(f"start_step {start_step} outside [0,{num_sampling_steps})")
    if device is None:
        try:
            device = next(diffusion.parameters()).device
        except (AttributeError, StopIteration):
            device = pre_state.device
    if atom_mask.dim() == 1:
        atom_mask = atom_mask.unsqueeze(0)
    atom_mask = atom_mask.to(device).repeat_interleave(multiplicity, 0)
    shape = (*atom_mask.shape, 3)

    # captured conditioning stores partials as {"__partial__": ...}: reuse the
    # shared move_conditioning so callables and tensors are rebuilt correctly
    from ..native_atom14.dpo_trainer import move_conditioning
    network_condition_kwargs = move_conditioning(network_condition_kwargs,
                                                 device=device)

    sigmas, gammas, step_scales, noise_scales = schedule_arrays(
        diffusion, num_sampling_steps, device=device,
        step_scale=step_scale, noise_scale=noise_scale)
    # exact official zip order: (sigma_t-1, sigma_t, gamma_t, step, noise)
    pairs = list(zip(sigmas[:-1], sigmas[1:], gammas[1:],
                     step_scales, noise_scales))

    # §16: one group seed makes the K-sibling continuation reproducible
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    # §14: all K siblings start from the identical state
    atom_coords = pre_state.detach().float().to(device).unsqueeze(0).repeat(
        multiplicity, 1, 1)
    if assert_synchronized and multiplicity > 1:
        synced = torch.equal(atom_coords, atom_coords[0:1].expand_as(atom_coords))
        if not synced:
            raise RuntimeError("sibling pre-states are not identical")

    center, compute_random_augmentation, weighted_rigid_align = _boltz_helpers()

    first_query = first_anchor = None
    first_sigma = None
    coords_traj_tail = []
    for step_idx in range(start_step, num_sampling_steps):
        sigma_tm, sigma_t, gamma, step_scale_t, noise_scale_t = pairs[step_idx]
        sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()
        # sigma_tm is sigma_t-1 and sigma_t is sigma_t
        t_hat = sigma_tm * (1 + gamma)
        noise_var = noise_scale_t ** 2 * (t_hat ** 2 - sigma_tm ** 2)

        atom_coords = center(atom_coords, atom_mask)
        if diffusion.coordinate_augmentation_inference:
            random_R, random_tr = compute_random_augmentation(
                multiplicity, device=atom_coords.device, dtype=atom_coords.dtype)
            atom_coords = (torch.einsum("bmd,bds->bms", atom_coords, random_R)
                           + random_tr)
        eps = noise_scale_t * sqrt(noise_var) * torch.randn(
            shape, device=device, dtype=atom_coords.dtype)
        atom_coords_noisy = atom_coords + eps
        atom_coords_denoised, _net_out = diffusion.preconditioned_network_forward(
            atom_coords_noisy, t_hat, training=False,
            network_condition_kwargs=dict(multiplicity=multiplicity,
                                          **network_condition_kwargs))
        if diffusion.alignment_reverse_diff:
            with torch.autocast("cuda", enabled=False):
                atom_coords_noisy = weighted_rigid_align(
                    atom_coords_noisy.float(), atom_coords_denoised.float(),
                    atom_mask.float(), atom_mask.float())
            atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)
        denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
        atom_coords_next = (atom_coords_noisy
                            + step_scale_t * (sigma_t - t_hat) * denoised_over_sigma)
        if step_idx == start_step:
            first_query = atom_coords_noisy.detach().clone()
            first_anchor = atom_coords_denoised.detach().clone()
            first_sigma = float(t_hat)
        coords_traj_tail.append(atom_coords.detach().float().cpu().clone())
        atom_coords = atom_coords_next
    coords_traj_tail.append(atom_coords.detach().float().cpu().clone())

    return {
        "endpoint_coords": atom_coords,
        "first_query": first_query,
        "first_anchor": first_anchor,
        "first_sigma": first_sigma,
        "coords_traj_tail": coords_traj_tail,
    }
