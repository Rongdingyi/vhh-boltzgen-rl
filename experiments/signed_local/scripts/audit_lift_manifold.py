#!/usr/bin/env python
"""Lift manifold audit (task book §20-§24)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))
from vhh_rl.signed_local.lift_audit import run_manifold_audit  # noqa: E402
from vhh_rl.signed_local.trainer import _load_design_positions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=Path,
                        default=ROOT / "runs/signed_local/edges/train_edges.jsonl")
    parser.add_argument("--n-sigma", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/signed_local/manifold")
    args = parser.parse_args()
    design = _load_design_positions(ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl")
    summary = run_manifold_audit(
        Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt"),
        args.edges, args.out, design, n_sigma=args.n_sigma,
        rollout_root=ROOT / "runs/cf_opsd/rollouts/train",
        conditioning_dir=ROOT / "runs/native_pool/conditioning")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
