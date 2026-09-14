#!/usr/bin/env python3
"""Persistent scorer worker (task book §7.2).

stdin:  {"spec": {"sequence": str, "design_positions": [...],
                  "cdr1": [...], "cdr2": [...], "cdr3": [...], "chain_id": "A"},
         "sequences": [str, ...]}
stdout: {"scores": [float, ...]}

The spec field is required once per case batch (the scorer needs the CDR
regions to build VHHSpec); sequences are complete AA20 sequences aligned to
that case.  Runs in the vhh-guidance-esmc env; loads the frozen VHH-source
ensemble once, eval() + no_grad for the whole process lifetime.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch
import yaml


def build_spec(row: dict, spec_cls) -> object:
    sequence = str(row["sequence"]).strip().upper()
    design = tuple(sorted(int(i) for i in row["design_positions"]))
    chars = list(sequence)
    for pos in design:
        chars[pos] = "X"
    regions = {
        "cdr1": tuple(int(i) for i in row["cdr1"]),
        "cdr2": tuple(int(i) for i in row["cdr2"]),
        "cdr3": tuple(int(i) for i in row["cdr3"]),
    }
    # VHHSpec's JSON contract is 1-based (ProteinMPNN fixed_positions
    # convention); the RL manifest is 0-based.  Shifted here, at the boundary.
    return spec_cls.from_dict(
        {
            "chain_id": str(row.get("chain_id", "A")),
            "masked_sequence": "".join(chars),
            "design_positions": [i + 1 for i in design],
            "regions": {k: [i + 1 for i in v] for k, v in regions.items()},
            "scorer_regions": {k: [i + 1 for i in v] for k, v in regions.items()},
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer-root", required=True)
    parser.add_argument("--objective", default="native_probability")
    args = parser.parse_args()

    # The VHH-source ensemble loader prints progress to stdout, which would
    # corrupt the JSONL protocol.  Protocol lines go to the REAL stdout only;
    # everything else (imports, loader logs) is diverted to stderr.
    import io as _io
    protocol_out = sys.__stdout__
    sys.stdout = sys.stderr

    root = Path(args.scorer_root)
    sys.path.insert(0, str(root / "src"))
    from vhh_guidance.adapters.vhh_source_adapter import VHHSourceAdapter
    from vhh_guidance.schemas import VHHSpec

    config = yaml.safe_load((root / "configs" / "models.local.yaml").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    warnings.filterwarnings("ignore")
    scorer = VHHSourceAdapter(config, device)
    scorer.load()
    current_spec = None
    current_key = None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        spec_row = request["spec"]
        key = json.dumps(spec_row, sort_keys=True)
        if key != current_key:
            current_spec = build_spec(spec_row, VHHSpec)
            current_key = key
        sequences = request["sequences"]
        results = scorer.score_sequences(sequences, current_spec)
        scores = []
        guard = []
        for result in results:
            if args.objective == "native_probability":
                scores.append(float(result.camelid_native_likeness_score))
            elif args.objective == "nativeness_margin":
                scores.append(float(result.nativeness_margin))
            elif args.objective == "nativeness_cdr_margin":
                scores.append(float(result.nativeness_margin))
            elif args.objective == "cdr_margin_guarded":
                # Round-2: main reward = CDR camelid margin (unbounded logit
                # margin); the global nativeness probability rides along as
                # the guardrail signal.  The parent combines them.
                scores.append(float(result.cdr_camelid_margin))
                guard.append(float(result.camelid_native_likeness_score))
            else:
                raise SystemExit(f"unknown objective {args.objective}")
        reply = {"scores": scores}
        if guard:
            reply["guard"] = guard
        protocol_out.write(json.dumps(reply) + "\n")
        protocol_out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
