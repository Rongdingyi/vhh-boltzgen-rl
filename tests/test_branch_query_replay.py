"""Full-batch query replay semantics (task book §31/§72)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from _branch_fixtures import make_feats  # noqa: E402
from vhh_rl.branch_distill.query_fit import forward_peer_prediction  # noqa: E402


class FakeStructureModule:
    def __init__(self):
        self.seen = {}

    def preconditioned_network_forward(self, noisy, sigma, training=False,
                                       network_condition_kwargs=None):
        self.seen["shape"] = tuple(noisy.shape)
        self.seen["sigma_shape"] = tuple(sigma.shape)
        self.seen["multiplicity"] = network_condition_kwargs["multiplicity"]
        out = noisy.clone()
        out[0] += 10.0                                # mark branch index 0 only
        return out, {}


class FakeModel:
    def __init__(self):
        self.structure_module = FakeStructureModule()


def _conditioning():
    feats = make_feats()
    feats = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in feats.items()}
    return {"s_inputs": torch.zeros(1, 4, 1), "s_trunk": torch.zeros(1, 4, 1),
            "feats": feats, "diffusion_conditioning": {"nested": torch.zeros(1)}}


def test_replay_keeps_full_k_batch_and_selects_peer():
    k, n = 8, 17
    query = torch.zeros(k, n, 3)
    model = FakeModel()
    pred = forward_peer_prediction(model, full_query_batch=query, sigma=1.25,
                                   conditioning=_conditioning(), multiplicity=k,
                                   peer_index=5, device="cpu")
    assert model.structure_module.seen["shape"] == (k, n, 3)
    assert model.structure_module.seen["sigma_shape"] == (k,)
    assert model.structure_module.seen["multiplicity"] == k
    assert float(pred.mean()) == 0.0                  # peer slice 5, not branch 0


def test_multiplicity_mismatch_is_rejected():
    model = FakeModel()
    try:
        forward_peer_prediction(model, full_query_batch=torch.zeros(3, 17, 3),
                                sigma=1.0, conditioning=_conditioning(),
                                multiplicity=8, peer_index=0, device="cpu")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_network_kwargs_moves_nested_conditioning():
    from vhh_rl.branch_distill.query_fit import network_kwargs

    conditioning = {"s_inputs": torch.zeros(1), "s_trunk": torch.zeros(1),
                    "feats": {"coords": torch.zeros(2, 3)},
                    "diffusion_conditioning": {"nested": {"value": torch.zeros(1)}}}
    kwargs = network_kwargs(conditioning, 4, device="cpu")
    assert kwargs["multiplicity"] == 4
    assert kwargs["feats"]["coords"].device.type == "cpu"
    assert kwargs["diffusion_conditioning"]["nested"]["value"].device.type == "cpu"
