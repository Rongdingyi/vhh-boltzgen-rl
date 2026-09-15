#!/usr/bin/env python
"""CF-DPO v2 experiment 4 pilot: signed / v2 training + held-out evaluation."""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

BASE = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt")
GRAPH = ROOT / "runs/cf_dpo_v2/graph/pilot_graph.pt"
OUT = ROOT / "runs/cf_dpo_v2"
HELDOUT = ["sab2_4hf5_h", "sab2_4mwf_h", "sab2_5mp6_h", "sab2_6cvk_b2"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["signed", "v2"])
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    out_dir = OUT / f"{args.variant}_u{args.updates}"
    if not args.eval_only:
        from vhh_rl.cf_dpo_v2.signed_trainer import run_signed

        summary = run_signed(BASE, GRAPH, out_dir, variant=args.variant, tau=args.tau,
                             kappa=args.kappa, updates=args.updates, checkpoint_every=50,
                             seed=args.seed, log_tag=f"cfd2-{args.variant}")
        print(json.dumps(summary, indent=1))
    if args.eval or args.eval_only:
        from vhh_rl.cf_opsd.evaluator import evaluate
        for step in (50, 100):
            ckpt = out_dir / f"checkpoint_{step:04d}.pt"
            if ckpt.is_file():
                run_root = out_dir / f"eval_u{step}"
                shutil.rmtree(run_root, ignore_errors=True)  # fresh eval pools
                res = evaluate(HELDOUT, ckpt, f"cfd2_{args.variant}_u{step}",
                               run_root=run_root)
                print(f"eval u{step}: reward_mean={res['reward_mean']}")


if __name__ == "__main__":
    main()
