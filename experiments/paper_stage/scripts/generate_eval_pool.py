#!/usr/bin/env python
"""Second-reward pipeline 3/3a: generate an evaluation pool for a checkpoint.

Like scripts/native_eval.py but scores nothing; writes a JSONL of decoded
sequences (+ invalid/FR diagnostics) for the ESM-C scorer.
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

MANIFESTS = {
    "heldout": ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl",
    "valid100": ROOT / "runs/round1_rl_split/rl_manifest_valid100test.jsonl",
}


def load_cases(split: str):
    out = []
    for line in MANIFESTS[split].open():
        row = json.loads(line)
        if split == "heldout" and row.get("split") != "test":
            continue
        out.append(RLCase(
            case_id=row["case_id"], structure_path=Path(row["structure_path"]),
            chain_id=row.get("chain_id", "A"), full_sequence=row["full_sequence"],
            design_positions=tuple(row["design_positions"]),
            fr_positions=tuple(row["fr_positions"]), split=split,
            seed_base=int(row.get("seed_base", 0)),
        ))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--split", default="heldout", choices=list(MANIFESTS))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=700000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cases = [c for i, c in enumerate(load_cases(args.split)) if i % args.nshards == args.shard]
    kwargs = dict(sampling_steps=args.sampling_steps, diffusion_batch_size=args.num_samples)
    if args.ckpt is not None:
        kwargs["design_ckpt"] = args.ckpt
    adapter = NativeDesignAdapter(**kwargs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for case in cases:
            capture = adapter.generate(
                case.structure_path.parent / "design.yaml", case.case_id,
                ROOT / "runs/paper_stage/second_reward/gen" / args.tag / case.case_id,
                num_designs=args.num_samples, seed=case.seed_base + args.seed_offset,
            )
            for sample in capture.samples:
                fr = fr_check(sample.decoded_sequence, case.full_sequence, case.fr_positions)
                fh.write(json.dumps({
                    "key": f"{case.case_id}:{sample.sample_index}",
                    "case_id": case.case_id,
                    "sample_index": sample.sample_index,
                    "sequence": sample.decoded_sequence,
                    "contains_invalid": sample.contains_invalid,
                    "fr_mismatch": fr["fr_mismatch_count"],
                }) + "\n")
            print(f"[gen-{args.tag}] {case.case_id}: {len(capture.samples)} samples", flush=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
