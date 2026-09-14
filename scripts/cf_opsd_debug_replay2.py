#!/usr/bin/env python
"""Granular replay debug: sigma shape, eval mode, autocast dtype."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cf_opsd.checkpoint import load_base_model  # noqa: E402
from vhh_rl.cf_opsd.rollout import load_cond_kwargs, load_contexts, load_trajectories  # noqa: E402
from vhh_rl.native_atom14.dpo_trainer import move_conditioning  # noqa: E402

ROLL = ROOT / "runs/cf_opsd/rollouts/train"
BASE = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt")


def main() -> None:
    case_dir = sorted(ROLL.iterdir())[0]
    traj_file = sorted(case_dir.glob("seed*.pt"))[0]
    trajs = sorted(load_trajectories(traj_file), key=lambda t: t.sample_index)
    contexts = load_contexts(traj_file)
    stored = load_cond_kwargs(traj_file)
    kwargs = move_conditioning(stored)
    kwargs["multiplicity"] = int(stored["multiplicity"])
    step = 0
    ctx = contexts[step]
    q = ctx["query"].cuda().float()
    anchors = ctx["anchors"].cuda().float()
    sigma_scalar = float(trajs[0].sigmas[step])

    base = load_base_model(BASE, device="cuda")

    def run(tag, sigma, eval_mode, autocast_dtype):
        model = base
        if eval_mode:
            model.eval()
        if autocast_dtype is None:
            with torch.no_grad():
                out, _ = model.structure_module.preconditioned_network_forward(
                    q, sigma, training=False, network_condition_kwargs=kwargs)
        else:
            with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype, enabled=True):
                out, _ = model.structure_module.preconditioned_network_forward(
                    q, sigma, training=False, network_condition_kwargs=kwargs)
        diff = (out.float() - anchors).abs()
        print(f"{tag}: maxdiff={diff.max().item():.3e} per-elem-max="
              f"{[round(diff[i].max().item(), 4) for i in range(min(4, diff.shape[0]))]}")

    print(f"anchor abs max = {anchors.abs().max().item():.3e}")
    for prec in ("highest", "high", "medium"):
        torch.set_float32_matmul_precision(prec)
        with torch.no_grad():
            out, _ = base.structure_module.preconditioned_network_forward(
                q, torch.full((q.shape[0],), sigma_scalar, device="cuda"),
                training=False, network_condition_kwargs=kwargs)
        diff = (out.float() - anchors).abs().max().item()
        rel = diff / anchors.abs().max().item()
        print(f"matmul_precision={prec}: maxdiff={diff:.3e} rel={rel:.3e}")


if __name__ == "__main__":
    main()
