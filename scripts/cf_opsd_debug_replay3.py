#!/usr/bin/env python
"""Decisive replay debug: per-step kwargs vs step-0 kwargs at each candidate step."""
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
    base = load_base_model(BASE, device="cuda")
    steps = sorted(contexts)
    print("captured steps:", steps)
    for step in steps:
        ctx = contexts[step]
        sig = float(trajs[0].sigmas[step])
        for tag, ck in (("own", load_cond_kwargs(traj_file, step=step)),
                        ("step0", load_cond_kwargs(traj_file, step=0))):
            kw = move_conditioning(ck)
            kw["multiplicity"] = int(ck.get("multiplicity", 1))
            with torch.no_grad():
                out, _ = base.structure_module.preconditioned_network_forward(
                    ctx["query"].cuda().float(), torch.full((ctx["query"].shape[0],), sig, device="cuda"),
                    training=False, network_condition_kwargs=kw)
            d = (out.float().cpu() - ctx["anchors"]).abs()
            print(f"step={step} kwargs={tag}: maxdiff={d.max().item():.3e} "
                  f"sigma={sig:.3f} qmax={ctx['query'].abs().max().item():.1f} "
                  f"amax={ctx['anchors'].abs().max().item():.1f}")


if __name__ == "__main__":
    main()
