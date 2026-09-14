"""Smoke: native design capture + decode parity + FR check (task book §64).

Uses 2 train cases and 2 held-out cases by default, 4 samples each.
Checks:
  1. capture works and official writer ran
  2. path A (official writer sequence captured at res_from_atom14 call) vs
     path B (re-decode from saved tensors via the same official function)
  3. FR hard check vs the manifest reference
  4. UNK accounting
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.native_atom14.adapter import NativeDesignAdapter  # noqa: E402
from vhh_rl.native_atom14.decode import decode_atom14, fr_check, sequence_from_feat  # noqa: E402
from vhh_rl.data.case import RLCase  # noqa: E402


def load_manifest(path: Path) -> dict[str, RLCase]:
    cases = {}
    for line in path.open():
        row = json.loads(line)
        cases[row["case_id"]] = RLCase(
            case_id=row["case_id"],
            structure_path=Path(row["structure_path"]),
            chain_id=row.get("chain_id", "A"),
            full_sequence=row["full_sequence"],
            design_positions=tuple(row["design_positions"]),
            fr_positions=tuple(row["fr_positions"]),
            split=row.get("split", "train"),
            seed_base=int(row.get("seed_base", 0)),
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs/native_smoke")
    parser.add_argument("--samples-per-case", type=int, default=4)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--diffusion-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()

    split_cases = load_manifest(ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl")
    train = [c for c in split_cases.values() if c.split == "train"][:2]
    heldout = [c for c in split_cases.values() if c.split == "test"][:2]
    cases = train + heldout
    assert len(cases) == 4, "smoke expects 2 train + 2 held-out"

    adapter = NativeDesignAdapter(
        sampling_steps=args.sampling_steps,
        diffusion_batch_size=args.diffusion_batch_size,
    )
    print(f"[smoke] checkpoint sha256={adapter.checkpoint_sha256}")

    report = {"cases": {}, "pass": True}
    n_checked = 0
    for case in cases:
        spec = case.structure_path.parent / "design.yaml"
        assert spec.is_file(), spec
        run_root = args.out / "runs" / case.case_id
        capture = adapter.generate(
            spec, case.case_id, run_root,
            num_designs=args.samples_per_case, seed=args.seed,
        )
        print(f"[smoke] {case.case_id}: {len(capture.samples)} samples in {capture.elapsed_seconds:.0f}s")
        case_report = {"samples": [], "decode_parity": True, "fr_mismatch": 0, "n_unk": 0}

        for sample in capture.samples:
            # path B: re-decode from captured tensors (no reference to writer state)
            feat_copy = {k: v.clone() for k, v in sample.feats_common.items()}
            feat_copy["coords"] = sample.coords.clone()
            out_b = decode_atom14(feat_copy)
            seq_b, _, invalid_b = sequence_from_feat(out_b)
            parity = seq_b == sample.decoded_sequence
            fr = fr_check(seq_b, case.full_sequence, case.fr_positions)
            if not parity:
                case_report["decode_parity"] = False
                report["pass"] = False
            if fr["fr_mismatch_count"] > 0:
                case_report["fr_mismatch"] += fr["fr_mismatch_count"]
                report["pass"] = False
            if invalid_b:
                case_report["n_unk"] += 1
            case_report["samples"].append({
                "sample_index": sample.sample_index,
                "decode_parity": parity,
                "fr_mismatch": fr["fr_mismatch_count"],
                "contains_unk": invalid_b,
                "seq_len": len(seq_b),
            })
            n_checked += 1
        report["cases"][case.case_id] = case_report

    report["n_checked"] = n_checked
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "smoke_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print("SMOKE PASS" if report["pass"] else "SMOKE FAIL")


if __name__ == "__main__":
    main()
