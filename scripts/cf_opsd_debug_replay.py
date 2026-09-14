#!/usr/bin/env python
"""Debug: which conditioning source reproduces the stored sampling anchor?"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cf_opsd.checkpoint import load_base_model  # noqa: E402
from vhh_rl.cf_opsd.rollout import load_contexts, load_trajectories  # noqa: E402
from vhh_rl.native_atom14.dpo_trainer import load_conditioning, move_conditioning  # noqa: E402

ROLL = ROOT / "runs/cf_opsd/rollouts/train"
BASE = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt")


def main() -> None:
    case_dir = sorted(ROLL.iterdir())[0]
    traj_file = sorted(case_dir.glob("seed*.pt"))[0]
    trajs = sorted(load_trajectories(traj_file), key=lambda t: t.sample_index)
    contexts = load_contexts(traj_file)
    case_id = case_dir.name
    step = len(trajs[0].query_states) // 2
    ctx = contexts[step]
    print(f"case={case_id} step={step} mult={ctx['multiplicity']} "
          f"query={tuple(ctx['query'].shape)} anchors={tuple(ctx['anchors'].shape)}")
    print(f"writer feats_common atoms={trajs[0].feats_common['atom_pad_mask'].numel()}")

    base = load_base_model(BASE, device="cuda")
    sigma = torch.tensor([trajs[0].sigmas[step]], device="cuda")

    def replay(kwargs, query):
        with torch.no_grad():
            den, _ = base.structure_module.preconditioned_network_forward(
                query.cuda().float(), sigma, training=False, network_condition_kwargs=kwargs)
        return den.float().cpu()

    # A: conditioning cache (captured with num_designs=1), full-context batch, recorded mult
    cond = move_conditioning(load_conditioning(ROOT / "runs/native_pool/conditioning", case_id))
    kwargs_a = {"s_inputs": cond["s_inputs"], "s_trunk": cond["s_trunk"],
                "feats": cond["feats"], "multiplicity": ctx["multiplicity"],
                "diffusion_conditioning": cond["diffusion_conditioning"]}
    out_a = replay(kwargs_a, ctx["query"])
    print(f"A cache-feats + mult={ctx['multiplicity']}: maxdiff="
          f"{(out_a - ctx['anchors']).abs().max().item():.3e} "
          f"feat_atoms={cond['feats']['atom_pad_mask'].numel()}")

    # compare cache feats vs writer feats key shapes/masks
    keys = ["atom_to_token", "ref_pos", "atom_pad_mask", "design_mask", "fake_atom_mask"]
    for k in keys:
        a = cond["feats"].get(k) if isinstance(cond["feats"], dict) else None
        b = trajs[0].feats_common.get(k)
        sa = tuple(a.shape) if torch.is_tensor(a) else None
        sb = tuple(b.shape) if torch.is_tensor(b) else None
        ha = int(a.float().sum().item() * 1000) if torch.is_tensor(a) else None
        hb = int(b.float().sum().item() * 1000) if torch.is_tensor(b) else None
        print(f"key {k}: cache={sa} sum*1e3={ha} | writer={sb} sum*1e3={hb}")

    # B3: writer feats with batch dim restored + recorded multiplicity + full query
    feats_b = {}
    for k, v in trajs[0].feats_common.items():
        if torch.is_tensor(v) and k != "coords":
            feats_b[k] = v.unsqueeze(0)
        else:
            feats_b[k] = v
    feats_b["coords"] = ctx["query"].clone()
    kwargs_b = {"s_inputs": cond["s_inputs"], "s_trunk": cond["s_trunk"],
                "feats": feats_b, "multiplicity": ctx["multiplicity"],
                "diffusion_conditioning": cond["diffusion_conditioning"]}
    out_b = replay(kwargs_b, ctx["query"])
    print(f"B3 writer-feats(batched) + mult={ctx['multiplicity']}: maxdiff="
          f"{(out_b - ctx['anchors']).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
