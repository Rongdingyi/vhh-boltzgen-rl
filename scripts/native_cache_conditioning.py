#!/usr/bin/env python
"""Cache per-case trunk conditioning for DPO training (frozen trunk outputs).

For each case: run the official design CLI in-process with a hook on
``AtomDiffusion.sample`` that grabs s_inputs / s_trunk / feats /
diffusion_conditioning and aborts before denoising.

Usage: native_cache_conditioning.py --split train [--shard i --nshards N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.data.case import RLCase  # noqa: E402
from vhh_rl.native_atom14.adapter import NativeDesignAdapter  # noqa: E402

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
    parser.add_argument("--split", required=True, choices=list(MANIFESTS))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/native_pool/conditioning")
    parser.add_argument("--seed-offset", type=int, default=700000)
    args = parser.parse_args()

    cases = [c for i, c in enumerate(load_cases(args.split)) if i % args.nshards == args.shard]
    adapter = NativeDesignAdapter(sampling_steps=50, diffusion_batch_size=1)
    args.out.mkdir(parents=True, exist_ok=True)
    for case in cases:
        out_path = args.out / f"{case.case_id}.pt"
        if out_path.is_file():
            print(f"[cond] {case.case_id}: exists, skip")
            continue
        spec = case.structure_path.parent / "design.yaml"
        try:
            payload = adapter.capture_conditioning(
                spec, case.case_id,
                run_root=ROOT / "runs/native_cache" / case.case_id,
                seed=case.seed_base + args.seed_offset,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[cond] {case.case_id}: FAILED {exc}")
            continue
        torch_payload = {
            "s_inputs": payload["s_inputs"],
            "s_trunk": payload["s_trunk"],
            "feats": payload["feats"],
            "diffusion_conditioning": payload["diffusion_conditioning"],
            "case_id": case.case_id,
            "checkpoint_sha256": payload["checkpoint_sha256"],
        }
        tmp = out_path.with_suffix(".pt.tmp")
        torch.save(torch_payload, tmp)
        tmp.replace(out_path)
        print(f"[cond] {case.case_id}: cached -> {out_path}")


if __name__ == "__main__":
    main()
