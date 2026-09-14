#!/usr/bin/env python
"""Phase B/D: generate a native atom14 pool + score rewards (task book §14-19).

Usage:
  native_generate_pool.py --split train   --num-samples 32 [--shard i --nshards N]
  native_generate_pool.py --split heldout --num-samples 8
  native_generate_pool.py --split valid100 --num-samples 8

For each assigned case:
  1. run the official native design CLI (adapter, seeded, capture hook)
  2. FR hard-check decoded sequences
  3. score decoded sequences with the existing CDR classifier (cdr_camelid_margin)
  4. save pool (coords/feats/metadata/cif/fasta)
Writes a shard summary JSON with invalid/FR/reward stats.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.data.case import RLCase  # noqa: E402
from vhh_rl.native_atom14.adapter import NativeDesignAdapter  # noqa: E402
from vhh_rl.native_atom14.decode import fr_check  # noqa: E402
from vhh_rl.native_atom14.reward import make_reward_adapter, score_case_sequences  # noqa: E402
from vhh_rl.native_atom14.sample_dataset import save_case_pool  # noqa: E402

MANIFESTS = {
    "train": ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl",
    "heldout": ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl",
    "valid100": ROOT / "runs/round1_rl_split/rl_manifest_valid100test.jsonl",
}


def load_cases(split: str) -> list[RLCase]:
    out = []
    for line in MANIFESTS[split].open():
        row = json.loads(line)
        row_split = row.get("split", "train")
        if split == "train" and row_split != "train":
            continue
        if split == "heldout" and row_split != "test":
            continue
        out.append(
            RLCase(
                case_id=row["case_id"],
                structure_path=Path(row["structure_path"]),
                chain_id=row.get("chain_id", "A"),
                full_sequence=row["full_sequence"],
                design_positions=tuple(row["design_positions"]),
                fr_positions=tuple(row["fr_positions"]),
                split=split,
                seed_base=int(row.get("seed_base", 0)),
            )
        )
    return out


def git_commit() -> str:
    return subprocess.run(
        ["git", "-C", "/share/home/rongdingyi/programs/proteingen/boltzgen", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=list(MANIFESTS))
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--diffusion-batch-size", type=int, default=8)
    parser.add_argument("--pool-root", type=Path, default=ROOT / "runs/native_pool")
    parser.add_argument("--design-ckpt", type=Path, default=None,
                        help="checkpoint for the design model (default: official base)")
    parser.add_argument("--seed-offset", type=int, default=700000)
    parser.add_argument("--case-ids", nargs="*", default=None)
    args = parser.parse_args()

    cases = [c for i, c in enumerate(load_cases(args.split)) if i % args.nshards == args.shard]
    if args.case_ids:
        wanted = set(args.case_ids)
        cases = [c for c in cases if c.case_id in wanted] or [c for c in load_cases(args.split) if c.case_id in wanted]
    print(f"[pool] split={args.split} shard={args.shard}/{args.nshards} cases={len(cases)}")

    adapter_kwargs = dict(
        sampling_steps=args.sampling_steps,
        diffusion_batch_size=args.diffusion_batch_size,
    )
    if args.design_ckpt is not None:
        adapter_kwargs["design_ckpt"] = args.design_ckpt
    adapter = NativeDesignAdapter(**adapter_kwargs)
    reward = make_reward_adapter(cache_path=args.pool_root / "reward_cache.sqlite3")
    commit = git_commit()

    summary = {"split": args.split, "shard": args.shard, "nshards": args.nshards,
               "checkpoint_sha256": adapter.checkpoint_sha256, "boltzgen_commit": commit,
               "cases": {}}
    for case in cases:
        spec = case.structure_path.parent / "design.yaml"
        seed = case.seed_base + args.seed_offset
        run_root = args.pool_root / "runs" / args.split / case.case_id
        capture = adapter.generate(
            spec, case.case_id, run_root,
            num_designs=args.num_samples, seed=seed,
        )
        rows = []
        sequences = []
        for sample in capture.samples:
            fr = fr_check(sample.decoded_sequence, case.full_sequence, case.fr_positions)
            # path B: independent re-decode from the captured tensors (official fn)
            feat_copy = {k: v.clone() for k, v in sample.feats_common.items()}
            feat_copy["coords"] = sample.coords.clone()
            from vhh_rl.native_atom14.decode import decode_atom14, sequence_from_feat
            out_b = decode_atom14(feat_copy)
            seq_b, _, _ = sequence_from_feat(out_b)
            rows.append({
                "decoded_sequence": sample.decoded_sequence,
                "contains_invalid": sample.contains_invalid,
                "fr_mismatch": fr["fr_mismatch_count"],
                "decode_parity": seq_b == sample.decoded_sequence,
                "reward_raw": None,
            })
            sequences.append(sample.decoded_sequence)
        # reward: score only valid, FR-clean samples
        score_idx = [
            i for i, r in enumerate(rows)
            if not r["contains_invalid"] and r["fr_mismatch"] == 0
        ]
        if score_idx:
            batch = score_case_sequences(reward, case, [sequences[i] for i in score_idx])
            values = batch.raw_scores.tolist()
            for i, value in zip(score_idx, values):
                rows[i]["reward_raw"] = float(value)
        save_case_pool(
            pool_root=args.pool_root,
            split=args.split,
            case_id=case.case_id,
            capture=capture,
            sequence_rows=rows,
            reference_sequence=case.full_sequence,
            fr_positions=case.fr_positions,
            design_positions=case.design_positions,
            checkpoint_sha256=adapter.checkpoint_sha256,
            boltzgen_commit=commit,
            extra_meta={"pool_seed": seed, "sampling_steps": args.sampling_steps},
        )
        scored = [r["reward_raw"] for r in rows if r["reward_raw"] is not None]
        summary["cases"][case.case_id] = {
            "n_samples": len(rows),
            "n_scored": len(scored),
            "n_invalid": sum(1 for r in rows if r["contains_invalid"]),
            "n_fr_mismatch": sum(1 for r in rows if r["fr_mismatch"] > 0),
            "reward_mean": sum(scored) / len(scored) if scored else None,
            "elapsed_seconds": capture.elapsed_seconds,
        }
        case_summary = summary["cases"][case.case_id]
        print(f"[pool] {case.case_id}: n={len(rows)} scored={len(scored)} "
              f"invalid={case_summary['n_invalid']} "
              f"fr_mismatch={case_summary['n_fr_mismatch']} "
              f"reward_mean={case_summary['reward_mean']}")
        case_summary["checkpoint_sha256"] = adapter.checkpoint_sha256

    summary_path = args.pool_root / f"pool_summary_{args.split}_shard{args.shard}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=1))
    print(f"[pool] summary -> {summary_path}")


if __name__ == "__main__":
    main()
