#!/usr/bin/env python
"""Evaluate a native checkpoint arm: generate pool + score reward + geometry.

Usage:
  native_eval.py --arm n2 --ckpt runs/native_n2/checkpoint_0500.pt \
      --split heldout --num-samples 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.native_atom14.adapter import NativeDesignAdapter  # noqa: E402
from vhh_rl.native_atom14.decode import fr_check  # noqa: E402
from vhh_rl.native_atom14.reward import make_reward_adapter, score_case_sequences  # noqa: E402
from vhh_rl.native_atom14.sample_dataset import save_case_pool  # noqa: E402

MANIFESTS = {
    "heldout": ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl",
    "valid100": ROOT / "runs/round1_rl_split/rl_manifest_valid100test.jsonl",
}


def load_cases(split: str):
    from vhh_rl.data.case import RLCase
    out = []
    for line in MANIFESTS[split].open():
        row = json.loads(line)
        row_split = row.get("split", "train")
        if split == "heldout" and row_split != "test":
            continue
        out.append(RLCase(
            case_id=row["case_id"],
            structure_path=Path(row["structure_path"]),
            chain_id=row.get("chain_id", "A"),
            full_sequence=row["full_sequence"],
            design_positions=tuple(row["design_positions"]),
            fr_positions=tuple(row["fr_positions"]),
            split=split,
            seed_base=int(row.get("seed_base", 0)),
        ))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--split", required=True, choices=list(MANIFESTS))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--diffusion-batch-size", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=700000)
    parser.add_argument("--pool-root", type=Path, default=None)
    args = parser.parse_args()

    pool_root = args.pool_root or (ROOT / "runs/native_pool")
    cases = [c for i, c in enumerate(load_cases(args.split)) if i % args.nshards == args.shard]
    kwargs = dict(sampling_steps=args.sampling_steps,
                  diffusion_batch_size=args.diffusion_batch_size)
    if args.ckpt is not None:
        kwargs["design_ckpt"] = args.ckpt
    adapter = NativeDesignAdapter(**kwargs)
    reward = make_reward_adapter(cache_path=pool_root / "reward_cache.sqlite3")
    import subprocess
    commit = subprocess.run(
        ["git", "-C", "/share/home/rongdingyi/programs/proteingen/boltzgen", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()

    summary = {"arm": args.arm, "split": args.split, "shard": args.shard,
               "nshards": args.nshards, "checkpoint": str(args.ckpt) if args.ckpt else "base",
               "checkpoint_sha256": adapter.checkpoint_sha256, "boltzgen_commit": commit,
               "cases": {}}
    ckpt_tag = ""
    if args.ckpt is not None:
        import re
        m = re.search(r"checkpoint_(\d+)\.pt", str(args.ckpt))
        ckpt_tag = f"_s{m.group(1)}" if m else "_ckpt"
    split_name = f"{args.split}_{args.arm}{ckpt_tag}"
    for case in cases:
        spec = case.structure_path.parent / "design.yaml"
        seed = case.seed_base + args.seed_offset
        run_root = pool_root / "runs" / split_name / case.case_id
        capture = adapter.generate(spec, case.case_id, run_root,
                                   num_designs=args.num_samples, seed=seed)
        rows = []
        for sample in capture.samples:
            fr = fr_check(sample.decoded_sequence, case.full_sequence, case.fr_positions)
            rows.append({
                "decoded_sequence": sample.decoded_sequence,
                "contains_invalid": sample.contains_invalid,
                "fr_mismatch": fr["fr_mismatch_count"],
                "reward_raw": None,
            })
        score_idx = [i for i, r in enumerate(rows)
                     if not r["contains_invalid"] and r["fr_mismatch"] == 0]
        if score_idx:
            batch = score_case_sequences(
                reward, case, [rows[i]["decoded_sequence"] for i in score_idx])
            for i, value in zip(score_idx, batch.raw_scores.tolist()):
                rows[i]["reward_raw"] = float(value)
        save_case_pool(
            pool_root=pool_root, split=split_name, case_id=case.case_id,
            capture=capture, sequence_rows=rows,
            reference_sequence=case.full_sequence,
            fr_positions=case.fr_positions, design_positions=case.design_positions,
            checkpoint_sha256=adapter.checkpoint_sha256, boltzgen_commit=commit,
            extra_meta={"arm": args.arm, "sampling_steps": args.sampling_steps},
        )
        scored = [r["reward_raw"] for r in rows if r["reward_raw"] is not None]
        summary["cases"][case.case_id] = {
            "n_samples": len(rows), "n_scored": len(scored),
            "n_invalid": sum(1 for r in rows if r["contains_invalid"]),
            "n_fr_mismatch": sum(1 for r in rows if r["fr_mismatch"] > 0),
            "reward_mean": sum(scored) / len(scored) if scored else None,
        }
        print(f"[eval-{args.arm}] {case.case_id}: n={len(rows)} scored={len(scored)} "
              f"mean={summary['cases'][case.case_id]['reward_mean']}", flush=True)

    summary_path = pool_root / f"eval_summary_{args.arm}{ckpt_tag}_{args.split}_shard{args.shard}.json"
    summary_path.write_text(json.dumps(summary, indent=1))
    print(f"[eval-{args.arm}] summary -> {summary_path}")


if __name__ == "__main__":
    main()
