"""z(policy=ref)=0 identity and local DPO semantics (task book §76).

Heavy: needs CUDA + boltzgen checkpoint + built edge dataset; skipped otherwise.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
pytest.importorskip("boltzgen")

from vhh_rl.native_atom14.checkpoint import load_base_model, make_policy_reference  # noqa: E402
from vhh_rl.native_atom14.dpo_trainer import load_case_conditioning, move_conditioning  # noqa: E402
from vhh_rl.native_atom14.global_cf_step import signed_local_dpo_step  # noqa: E402
from vhh_rl.signed_local.edge_validator import load_edges  # noqa: E402
from vhh_rl.signed_local.local_dpo import signed_local_dpo_loss  # noqa: E402
from vhh_rl.signed_local.trainer import _load_design_positions  # noqa: E402

BASE = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt")
EDGES = ROOT / "runs/signed_local/edges/train_edges.jsonl"
CONTEXT = ROOT / "runs/signed_local/frozen_context/train_edges_context.jsonl"


@cuda
def test_z_zero_when_policy_equals_reference():
    if not EDGES.is_file():
        pytest.skip("edge dataset not built")
    if not BASE.is_file():
        pytest.skip("base checkpoint unavailable on this host")
    edges = load_edges(EDGES)[:8]
    design = _load_design_positions(ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl")
    model = load_base_model(BASE, device="cuda")
    policy, reference = make_policy_reference(model, device="cuda")
    for e in edges:
        cond = move_conditioning(load_case_conditioning(
            e.case_id, ROOT / "runs/native_pool/conditioning",
            ROOT / "runs/cf_opsd/rollouts/train"), device="cuda")
        feats = cond["feats"]
        kwargs = {"s_inputs": cond["s_inputs"], "s_trunk": cond["s_trunk"],
                  "feats": feats, "multiplicity": 1,
                  "diffusion_conditioning": cond["diffusion_conditioning"]}
        from vhh_rl.signed_local.local_mask import target_residue_mask

        mask = target_residue_mask(feats, e.position, design[e.case_id]).to("cuda")
        preferred, rejected = _EdgePair.of(e)
        out = signed_local_dpo_step(policy.structure_module, reference.structure_module,
                                    feats, preferred, rejected, mask, kwargs, beta=10.0)
        assert abs(float(out.dpo.z.mean())) < 1e-6, e.edge_id
        assert float(out.loss) == pytest.approx(float(torch.log(torch.tensor(2.0))), abs=1e-5)


class _EdgePair:
    @staticmethod
    def of(edge):
        anchor = torch.load(edge.anchor_coords_path, map_location="cuda",
                            weights_only=True).float()
        cf = torch.load(edge.cf_coords_path, map_location="cuda", weights_only=True).float()
        return (cf, anchor) if edge.preferred_side == "cf" else (anchor, cf)
