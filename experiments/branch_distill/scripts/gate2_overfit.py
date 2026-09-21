#!/usr/bin/env python
"""Gate 2: same-query replay + one-record overfit + downstream replay (§31-§39)."""
from __future__ import annotations

import argparse
import csv
import json
import sys

import torch

import _common as C  # noqa: E402
from vhh_rl.branch_distill import gates  # noqa: E402
from vhh_rl.branch_distill.decode import decode_coords_with_fr
from vhh_rl.branch_distill.local_target import changed_design_positions
from vhh_rl.branch_distill.metrics import design_hamming
from vhh_rl.branch_distill.query_fit import (
    fit_record, forward_peer_prediction, masked_mse, record_target_mask,
)
from vhh_rl.branch_distill.tail_sampler import continue_from_state
from vhh_rl.branch_distill.types import DistillRecord


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--override-gate", default=None)
    parser.add_argument("--updates", type=int, default=40)
    args = parser.parse_args()
    C.require_gate(C.GATE1_DIR / "gate1.json", "pass", True, args.override_gate, "Gate 1")
    C.GATE2_DIR.mkdir(parents=True, exist_ok=True)
    records = [DistillRecord(**{k: v for k, v in row.items()
                                if k in DistillRecord.__dataclass_fields__})
               for row in torch.load(C.GATE2_DIR / "records.pt", map_location="cpu",
                                     weights_only=False)["records"]]
    cases = C.load_cases(C.TRAIN_CASES)
    scorer = C.make_scorer()
    rows = []
    per_dir = C.GATE2_DIR / "per_record"
    for i, record in enumerate(records):
        case = cases[record.case_id]
        from vhh_rl.native_atom14.checkpoint import load_base_model
        base = load_base_model(C.BASE_CKPT, device=C.DEVICE)

        # A) same-query replay gate (student == behavior)
        pred = forward_peer_prediction(base, full_query_batch=record.full_query_batch,
                                       sigma=record.meta["sigma"],
                                       conditioning=record.conditioning,
                                       multiplicity=record.branch_count,
                                       peer_index=record.peer_index)
        replay_maxdiff = float((pred.cpu() - record.peer_anchor).abs().max())

        # C) one-record finite fit from a fresh base student
        result = fit_record(C.BASE_CKPT, record, updates=args.updates)
        student = result["student"]

        # D) downstream replay from the same pre_state/seed with the student
        from vhh_rl.branch_distill.query_fit import network_kwargs
        out = continue_from_state(
            student.structure_module,
            pre_state=record.pre_state.to(C.DEVICE),
            start_step=record.start_step,
            num_sampling_steps=C.SAMPLING_STEPS,
            multiplicity=record.branch_count,
            atom_mask=record.conditioning["feats"]["atom_pad_mask"].to(C.DEVICE),
            network_condition_kwargs=network_kwargs(
                record.conditioning, record.branch_count, device=C.DEVICE),
            seed=record.group_seed,
            step_scale=(record.meta.get("sampling_scales") or {}).get("step_scale"),
            noise_scale=(record.meta.get("sampling_scales") or {}).get("noise_scale"))
        peer_endpoint = out["endpoint_coords"][record.peer_index].detach().cpu()
        audit = decode_coords_with_fr(peer_endpoint, record.conditioning["feats"],
                                      case.full_sequence, case.fr_positions)
        reward = None
        if not audit["contains_invalid"] and audit["fr_mismatch"] == 0:
            batch = scorer.score_sequences([audit["sequence"]], spec=C.case_spec(case))
            reward = float(batch.raw_scores[0])
        changed = changed_design_positions(record.teacher_sequence,
                                           audit["sequence"], case.design_positions)
        hamming_after = design_hamming(record.teacher_sequence, audit["sequence"],
                                       case.design_positions)
        hamming_before = design_hamming(record.teacher_sequence, record.peer_sequence,
                                        case.design_positions)

        row = {
            "record_id": record.record_id, "case_id": record.case_id,
            "carrier_invalid": record.meta.get("carrier_invalid"),
            "carrier_fr": record.meta.get("carrier_fr"),
            "carrier_all_match": record.meta.get("carrier_all_match"),
            "carrier_verified_positions": len(record.meta.get("carrier_verified_positions", ())),
            "replay_maxdiff": replay_maxdiff,
            "initial_masked_mse": result["initial_masked_mse"],
            "final_masked_mse": result["final_masked_mse"],
            "mse_ratio": result["mse_ratio"],
            "teacher_reward": record.teacher_reward,
            "peer_reward": record.peer_reward,
            "reward_gap": record.reward_gap,
            "n_changed": len(record.changed_positions),
            "n_touched_atoms": int(record_target_mask(record).sum()),
            "postfit_reward": reward, "postfit_sequence": audit["sequence"],
            "hamming_before": hamming_before, "hamming_after": hamming_after,
            "teacher_match_after": audit["sequence"] == record.teacher_sequence,
            "invalid_after": audit["contains_invalid"],
            "fr_after": audit["fr_mismatch"],
            "param_drift": result["param_drift"],
        }
        rows.append(row)
        C.write_json(per_dir / f"record_{i:02d}.json", row)
        print(f"[gate2] {record.record_id}: replay={replay_maxdiff:.2e} "
              f"ratio={row['mse_ratio']:.3f} ham {hamming_before}->{hamming_after} "
              f"reward={reward}", flush=True)
    scorer.close()
    with (C.GATE2_DIR / "overfit_results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    verdict = gates.gate2_verdict(rows)
    payload = {"protocol_sha256": C.protocol_hash(), "n_records": len(rows),
               **verdict}
    C.write_json(C.GATE2_DIR / "gate2.json", payload)
    print(json.dumps(verdict, indent=1))
    if not verdict["pass"]:
        sys.exit(1)   # §39: STOP, no automatic method changes


if __name__ == "__main__":
    main()
