#!/usr/bin/env python
"""Frozen valid100 evaluation for a trained SL-CF-DPO checkpoint (§63)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))
from vhh_rl.signed_local.evaluator import evaluate_reward  # noqa: E402

MANIFEST = ROOT / "runs/round1_rl_split/rl_manifest_valid100test.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    args = parser.parse_args()
    cases = sorted(json.loads(l)["case_id"] for l in MANIFEST.open())
    res = evaluate_reward(args.ckpt, cases, args.tag, num_samples=args.num_samples,
                          run_root=ROOT / "runs/signed_local/valid100_eval" / args.tag,
                          manifest=MANIFEST)
    print(json.dumps({"tag": args.tag, "reward_mean": res["reward_mean"]}, indent=1))


if __name__ == "__main__":
    main()
