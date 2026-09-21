"""First resumed query semantics and K-sibling pre-state sync (task book §11/§14/§58)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.branch_distill import tail_sampler  # noqa: E402


class StepStub:
    def __init__(self):
        self.training = False
        self.step_scale_random = None
        self.step_scale_function = "constant"
        self.noise_scale_function = "constant"
        self.sampling_schedule = "af3"
        self.step_scale = 1.0
        self.noise_scale = 0.0          # deterministic continuation for the test
        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self.sigma_data = 16.0
        self.sigma_max = 160.0
        self.sigma_min = 4e-4
        self.rho = 7.0
        self.coordinate_augmentation_inference = False
        self.alignment_reverse_diff = False

    def sample_schedule_af3(self, num_sampling_steps=None):
        n = num_sampling_steps
        inv_rho = 1 / self.rho
        steps = torch.arange(n, dtype=torch.float32)
        sigmas = (self.sigma_max ** inv_rho
                  + steps / (n - 1) * (self.sigma_min ** inv_rho
                                       - self.sigma_max ** inv_rho)) ** self.rho
        sigmas = sigmas * self.sigma_data
        return torch.nn.functional.pad(sigmas, (0, 1), value=0.0)

    def sample_schedule_dilated(self, n=None):
        return self.sample_schedule_af3(n)

    def preconditioned_network_forward(self, noisy, sigma, training=False,
                                       network_condition_kwargs=None):
        return torch.zeros_like(noisy), {}


@pytest.fixture(autouse=True)
def patch_helpers(monkeypatch):
    monkeypatch.setattr(tail_sampler, "_boltz_helpers",
                        lambda: (_center, _augment, _align))


def _center(coords, mask):
    return coords - coords.mean(dim=1, keepdim=True)


def _augment(multiplicity, s_trans=1.0, device=None, dtype=None):
    return (torch.eye(3, device=device, dtype=dtype).expand(multiplicity, 3, 3),
            torch.zeros(multiplicity, 1, 3, device=device, dtype=dtype))


def _align(a, b, w, m):
    return a


def test_k_siblings_start_identical_and_shapes():
    n_atoms, k, steps = 5, 4, 10
    pre_state = torch.randn(n_atoms, 3)
    mask = torch.ones(1, n_atoms, dtype=torch.bool)
    out = tail_sampler.continue_from_state(
        StepStub(), pre_state=pre_state, start_step=3, num_sampling_steps=steps,
        multiplicity=k, atom_mask=mask,
        network_condition_kwargs={"feats": {}}, seed=7)
    assert out["endpoint_coords"].shape == (k, n_atoms, 3)
    assert out["first_query"].shape == (k, n_atoms, 3)
    assert out["first_anchor"].shape == (k, n_atoms, 3)
    assert isinstance(out["first_sigma"], float)


def test_first_query_is_not_the_state_but_a_noised_query():
    n_atoms, k = 6, 3
    pre_state = torch.arange(n_atoms * 3, dtype=torch.float32).reshape(n_atoms, 3)
    out = tail_sampler.continue_from_state(
        StepStub(), pre_state=pre_state, start_step=2, num_sampling_steps=8,
        multiplicity=k, atom_mask=torch.ones(1, n_atoms, dtype=torch.bool),
        network_condition_kwargs={"feats": {}}, seed=3)
    assert not torch.equal(out["first_query"][0], pre_state)
    assert out["first_sigma"] > 0


def test_invalid_start_step_is_rejected():
    with pytest.raises(ValueError):
        tail_sampler.continue_from_state(
            StepStub(), pre_state=torch.randn(4, 3), start_step=10,
            num_sampling_steps=10, multiplicity=2,
            atom_mask=torch.ones(1, 4, dtype=torch.bool),
            network_condition_kwargs={"feats": {}}, seed=1)


class AlignedStub(StepStub):
    def __init__(self):
        super().__init__()
        self.alignment_reverse_diff = True


def test_first_query_is_the_network_input_not_the_aligned_tensor(monkeypatch):
    """§15/§31: capture must happen before alignment_reverse_diff rewrites
    atom_coords_noisy, otherwise the stored query cannot replay the anchor."""
    seen_inputs = []

    class Stub(AlignedStub):
        def preconditioned_network_forward(self, noisy, sigma, training=False,
                                           network_condition_kwargs=None):
            seen_inputs.append(noisy.detach().clone())
            return noisy * 0.5, {}

    def align_adds_offset(a, b, w, m):
        return a + 100.0

    monkeypatch.setattr(tail_sampler, "_boltz_helpers",
                        lambda: (_center, _augment, align_adds_offset))
    out = tail_sampler.continue_from_state(
        Stub(), pre_state=torch.randn(5, 3), start_step=1, num_sampling_steps=5,
        multiplicity=1, atom_mask=torch.ones(1, 5, dtype=torch.bool),
        network_condition_kwargs={"feats": {}}, seed=5)
    assert torch.allclose(out["first_query"], seen_inputs[0], atol=1e-6)
    assert not torch.allclose(out["first_query"], seen_inputs[0] + 100.0, atol=1e-3)
