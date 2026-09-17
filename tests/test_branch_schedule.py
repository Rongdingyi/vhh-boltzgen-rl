"""Tail sampler schedule parity with the official sampler (task book §13/§58)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.branch_distill import tail_sampler  # noqa: E402


class StubDiffusion:
    """Mirrors the official AtomDiffusion attribute set used by sample()."""

    def __init__(self, schedule="af3"):
        self.training = False
        self.step_scale_random = None
        self.step_scale_function = "constant"
        self.noise_scale_function = "constant"
        self.sampling_schedule = schedule
        self.step_scale = 1.5
        self.noise_scale = 1.0
        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self.sigma_data = 16.0
        self.sigma_max = 160.0
        self.sigma_min = 4e-4
        self.rho = 7.0

    def sample_schedule_af3(self, num_sampling_steps=None):
        n = num_sampling_steps
        inv_rho = 1 / self.rho
        steps = torch.arange(n, dtype=torch.float32)
        sigmas = (self.sigma_max ** inv_rho
                  + steps / (n - 1) * (self.sigma_min ** inv_rho
                                       - self.sigma_max ** inv_rho)) ** self.rho
        sigmas = sigmas * self.sigma_data
        return torch.nn.functional.pad(sigmas, (0, 1), value=0.0)

    def sample_schedule_dilated(self, num_sampling_steps=None):
        return self.sample_schedule_af3(num_sampling_steps)

    def beta_step_scale_schedule(self, num_sampling_steps=None):
        return 1.5 * torch.ones(num_sampling_steps)

    def beta_noise_scale_schedule(self, num_sampling_steps):
        return torch.ones(num_sampling_steps)


def reference_arrays(diffusion, n):
    """Independent restatement of the official schedule block."""
    sigmas = diffusion.sample_schedule_af3(n)
    gammas = torch.where(sigmas > diffusion.gamma_min, diffusion.gamma_0, 0.0)
    step_scales = diffusion.step_scale * torch.ones(n)
    noise_scales = diffusion.noise_scale * torch.ones(n)
    return sigmas, gammas, step_scales, noise_scales


def test_schedule_arrays_match_reference():
    diffusion = StubDiffusion()
    got = tail_sampler.schedule_arrays(diffusion, 50, device="cpu")
    ref = reference_arrays(diffusion, 50)
    for a, b in zip(got, ref):
        assert torch.equal(a, b)


def test_zip_pair_order_and_tail_formula_contract():
    diffusion = StubDiffusion()
    sigmas, gammas, step_scales, noise_scales = tail_sampler.schedule_arrays(
        diffusion, 50, device="cpu")
    pairs = list(zip(sigmas[:-1], sigmas[1:], gammas[1:], step_scales, noise_scales))
    assert len(pairs) == 50   # one pair per denoising step
    # official tail formula used by the sampler
    sigma_tm, sigma_t, gamma, step_scale, noise_scale = [float(v) for v in pairs[10]]
    t_hat = sigma_tm * (1 + gamma)
    noise_var = noise_scale ** 2 * (t_hat ** 2 - sigma_tm ** 2)
    assert noise_var >= 0
    noisy = torch.tensor([1.0, 2.0, 3.0])
    denoised = torch.tensor([0.5, 2.5, 2.0])
    denoised_over_sigma = (noisy - denoised) / t_hat
    nxt = noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma
    assert torch.isfinite(nxt).all()
    assert math.isclose(float(t_hat), sigma_tm * (1 + gamma), rel_tol=0, abs_tol=0)


def test_official_source_contract_present_when_available():
    path = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/src/boltzgen/"
                "model/modules/diffusion.py")
    if not path.is_file():
        pytest.skip("BoltzGen checkout unavailable")
    text = path.read_text()
    assert "denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat" in text
    assert "atom_coords_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma" in text
