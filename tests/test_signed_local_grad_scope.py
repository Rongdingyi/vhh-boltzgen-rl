"""Local step gradient scope (task book §81): score_model only, ref frozen."""
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


@cuda
def test_only_policy_score_model_receives_gradients():
    if not EDGES.is_file() or not BASE.is_file():
        pytest.skip("edge dataset / checkpoint unavailable")
    edge = load_edges(EDGES)[0]
    design = _load_design_positions(ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl")
    model = load_base_model(BASE, device="cuda")
    policy, reference = make_policy_reference(model, device="cuda")
    cond = move_conditioning(load_case_conditioning(
        edge.case_id, ROOT / "runs/native_pool/conditioning",
        ROOT / "runs/cf_opsd/rollouts/train"), device="cuda")
    feats = cond["feats"]
    kwargs = {"s_inputs": cond["s_inputs"], "s_trunk": cond["s_trunk"],
              "feats": feats, "multiplicity": 1,
              "diffusion_conditioning": cond["diffusion_conditioning"]}
    mask = target_residue_mask(feats, edge.position, design[edge.case_id]).to("cuda")
    preferred = torch.load(edge.cf_coords_path, map_location="cuda",
                           weights_only=True).float()
    rejected = torch.load(edge.anchor_coords_path, map_location="cuda",
                          weights_only=True).float()
    params = trainable_score_params(policy)
    out = signed_local_dpo_step(policy.structure_module, reference.structure_module,
                                feats, preferred, rejected, mask, kwargs, beta=10.0)
    out.loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in params)
    for name, p in policy.named_parameters():
        if p.grad is not None:
            assert name.startswith("structure_module.score_model."), name
    for name, p in reference.named_parameters():
        assert p.grad is None, f"reference received grad: {name}"
