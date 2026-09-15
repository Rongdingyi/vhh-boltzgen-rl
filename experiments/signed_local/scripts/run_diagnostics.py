#!/usr/bin/env python
"""Gradient-direction + sigma robustness audits (task book §54-§56)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))
from vhh_rl.signed_local.diagnostics import (  # noqa: E402
    gradient_direction_audit, sigma_robustness_audit,
)

BASE = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt")
OUT = ROOT / "runs/signed_local/diagnostics"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=Path,
                        default=ROOT / "runs/signed_local/edges/train_edges.jsonl")
    parser.add_argument("--n-edges", type=int, default=32)
    parser.add_argument("--n-sigma", type=int, default=16)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    grad = gradient_direction_audit(BASE, args.edges, n_edges=args.n_edges,
                                    cond_dir=ROOT / "runs/native_pool/conditioning",
                                    rollout_root=ROOT / "runs/cf_opsd/rollouts/train")
    (OUT / "gradient_direction.json").write_text(json.dumps(grad, indent=1))
    sigma = sigma_robustness_audit(BASE, args.edges, n_edges=args.n_edges,
                                   n_sigma=args.n_sigma,
                                   cond_dir=ROOT / "runs/native_pool/conditioning",
                                   rollout_root=ROOT / "runs/cf_opsd/rollouts/train")
    (OUT / "sigma_robustness.json").write_text(json.dumps(sigma, indent=1))
    print(json.dumps({"gradient_gate": grad["gate"], "median_dz": grad["median_dz"],
                      "sigma_gate": sigma["gate"],
                      "median_positive_fraction": sigma["median_positive_fraction"]}, indent=1))


if __name__ == "__main__":
    main()
