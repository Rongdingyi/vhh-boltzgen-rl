#!/usr/bin/env python
"""Reproduce the exact test path: step=25, sigma[1] vs [B], fallback kwargs."""
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
    torch.set_float32_matmul_precision("high")
    case_dir = sorted(ROLL.iterdir())[0]
    traj_file = sorted(case_dir.glob("seed*.pt"))[0]
    trajs = sorted(load_trajectories(traj_file), key=lambda t: t.sample_index)
    contexts = load_contexts(traj_file)
    steps = sorted(contexts)
    step = steps[len(steps) // 2]
    ctx = contexts[step]
    stored_step = load_cond_kwargs(traj_file, step=step)
    print(f"case={case_dir.name} step={step} n_stored_steps={len(load_cond_kwargs(traj_file, step=None))}")
    base = load_base_model(BASE, device="cuda")
    for tag, sigma in (("sigma[1]", torch.tensor([float(trajs[0].sigmas[step])], device="cuda")),
                       ("sigma[B]", torch.full((ctx["query"].shape[0],), float(trajs[0].sigmas[step]), device="cuda"))):
        for kw_tag, ck in (("stored_step", stored_step), ("step0", load_cond_kwargs(traj_file))):
            kwargs = move_conditioning(ck)
            kwargs["multiplicity"] = int(ck.get("multiplicity", 1))
            with torch.no_grad():
                out, _ = base.structure_module.preconditioned_network_forward(
                    ctx["query"].cuda().float(), sigma, training=False,
                    network_condition_kwargs=kwargs)
            d = float((out.float().cpu() - ctx["anchors"]).abs().max())
            print(f"{tag} + {kw_tag}: maxdiff={d:.3e}")


if __name__ == "__main__":
    main()
