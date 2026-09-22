"""Branch-aware sigma sampling from the original training distribution (§18-§21).

The training noise law is the frozen BoltzGen one:
    sigma = sigma_data * exp(P_mean + P_std * z),  z ~ N(0, 1)
Conditional arms sample the same law restricted to sigma <= sigma_b (suffix)
or sigma > sigma_b (prefix) via the analytic normal CDF, so every arm keeps
100 *effective* optimizer updates (no ``g(sigma) * loss`` gating, §17).
"""
from __future__ import annotations

import math

import torch

from .step_randomness import make_generator

SQRT2 = math.sqrt(2.0)
REGIONS = ("full", "suffix", "prefix")


def z_boundary(structure_module, boundary_sigma: float) -> float:
    """y_b = (log(sigma_b/sigma_data) - P_mean) / P_std."""
    return ((math.log(float(boundary_sigma) / float(structure_module.sigma_data))
             - float(structure_module.P_mean)) / float(structure_module.P_std))


def phi(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / SQRT2))


def phi_inv(u: torch.Tensor) -> torch.Tensor:
    return SQRT2 * torch.erfinv(2.0 * u - 1.0)


def suffix_mass(structure_module, boundary_sigma: float) -> float:
    """q_b = P_train(sigma <= sigma_b)."""
    return float(phi(torch.tensor(z_boundary(structure_module, boundary_sigma))))


def sigma_from_z(structure_module, z: torch.Tensor) -> torch.Tensor:
    return structure_module.sigma_data * (
        structure_module.P_mean + structure_module.P_std * z).exp()


def sample_uniform(seed: int, device) -> torch.Tensor:
    g = make_generator(seed, device)
    u = torch.rand((), generator=g, device=device, dtype=torch.float32)
    return u.clamp_min(1e-7).clamp_max(1.0 - 1e-7)


def sample_region_sigma(structure_module, *, boundary_sigma: float,
                        region: str, seed: int, device) -> torch.Tensor:
    """Matched-quantile conditional sampling (§24).

    One uniform u per (seed, update) drives all three arms:
        full:   u
        suffix: u * q_b
        prefix: q_b + u * (1 - q_b)
    """
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}")
    device = device or getattr(structure_module, "device", None)
    u = sample_uniform(seed, device)
    if region == "full":
        z = phi_inv(u)
    else:
        q_b = float(suffix_mass(structure_module, boundary_sigma))
        if region == "suffix":
            z = phi_inv(u * q_b)
        else:  # prefix
            z = phi_inv(q_b + u * (1.0 - q_b))
    return sigma_from_z(structure_module, z).reshape(-1)
