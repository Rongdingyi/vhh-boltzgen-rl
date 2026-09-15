"""Preferred and rejected endpoints share one sigma/noise draw (task book §77).

With policy != reference, an identical-coordinate pair must still give z == 0:
any per-endpoint independent noising would break the cancellation.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
pytest.importorskip("boltzgen")

from vhh_rl.native_atom14.checkpoint import (  # noqa: E402
    load_base_model, make_policy_reference, trainable_score_params,
)
from vhh_rl.native_atom14.dpo_trainer import load_case_conditioning, move_conditioning  # noqa: E402
from vhh_rl.native_atom14.global_cf_step import signed_local_dpo_step  # noqa: E402
from vhh_rl.signed_local.edge_validator import load_edges  # noqa: E402
from vhh_rl.signed_local.local_mask import target_residue_mask  # noqa: E402
from vhh_rl.signed_local.trainer import _load_design_positions  # noqa: E402

BASE = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt")
EDGES = ROOT / "runs/signed_local/edges/train_edges.jsonl"
LOG2 = float(torch.log(torch.tensor(2.0)))


def _setup():
    if not EDGES.is_file() or not BASE.is_file():
        pytest.skip("edge dataset / checkpoint unavailable")
    edge = load_edges(EDGES)[0]
    design = _load_design_positions(ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl")
    model = load_base_model(BASE, device="cuda")
    policy, reference = make_policy_reference(model, device="cuda")
    with torch.no_grad():
        trainable_score_params(policy)[0].mul_(1.01)   # policy != reference
    cond = move_conditioning(load_case_conditioning(
        edge.case_id, ROOT / "runs/native_pool/conditioning",
        ROOT / "runs/cf_opsd/rollouts/train"), device="cuda")
    feats = cond["feats"]
    kwargs = {"s_inputs": cond["s_inputs"], "s_trunk": cond["s_trunk"],
              "feats": feats, "multiplicity": 1,
              "diffusion_conditioning": cond["diffusion_conditioning"]}
    mask = target_residue_mask(feats, edge.position, design[edge.case_id]).to("cuda")
    coords = torch.load(edge.anchor_coords_path, map_location="cuda",
                        weights_only=True).float()
    return policy, reference, feats, coords, mask, kwargs


@cuda
def test_identical_coords_give_zero_z():
    policy, reference, feats, coords, mask, kwargs = _setup()
    out = signed_local_dpo_step(policy.structure_module, reference.structure_module,
                                feats, coords, coords.clone(), mask, kwargs, beta=10.0)
    assert abs(float(out.dpo.z.mean())) < 1e-6
    assert float(out.loss) == pytest.approx(LOG2, abs=1e-5)


@cuda
def test_shared_sigma_noise_keeps_cancellation():
    policy, reference, feats, coords, mask, kwargs = _setup()
    sigma = policy.structure_module.noise_distribution(1)
    noise = torch.randn_like(coords.unsqueeze(0))
    out = signed_local_dpo_step(policy.structure_module, reference.structure_module,
                                feats, coords, coords.clone(), mask, kwargs,
                                beta=10.0, sigma=sigma, noise=noise)
    assert abs(float(out.dpo.z.mean())) < 1e-6
