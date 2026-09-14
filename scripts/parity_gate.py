#!/usr/bin/env python3
"""Gate 1 (rollout parity) + Gate 2 (replay logprob parity) in one job.

Gate 1: same seed -> original decoder.sample() sequence == rollout_with_logprobs
sequence, over 10 seeds per case.
Gate 2: same-policy replay action logprobs == rollout old_logprobs (fp32,
max_abs_diff <= 1e-5).

Writes gate JSON to the run dir; exits non-zero on any failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vhh_rl.adapters.boltzgen_if import BoltzGenIFAdapter  # noqa: E402
from vhh_rl.config import load_config  # noqa: E402
from vhh_rl.cli.common import capture_cases, manifest_cases  # noqa: E402


def main() -> int:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/round1_smoke.yaml")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("runs/round1/parity")
    out.mkdir(parents=True, exist_ok=True)
    cases = manifest_cases(cfg, splits=("train",))[: cfg.max_cases]
    adapter = BoltzGenIFAdapter(cfg.boltzgen_root, cfg.boltzgen_checkpoint, device="cuda")
    captured = capture_cases(adapter, cfg, cases, out / "captures")
    report = {"parity": [], "replay": []}
    ok = True
    for case, cap in zip(cases, captured):
        temperature = float(adapter.decoder.sampling_temperature)
        for seed in range(10):
            torch.manual_seed(seed)
            original = adapter.decoder.sample(
                s=cap.s, z=cap.z, edge_idx=cap.edge_idx, valid_mask=cap.valid_mask,
                feats=cap.feats,
            )
            original_seq = "".join(
                adapter.aa_order[int(i)]
                for i in (original["res_type"][0].argmax(dim=-1)[cap.valid_mask[0]] - adapter.offset).tolist()
            )
            traj = adapter.rollout_with_logprobs(cap, temperature, seed=seed)
            match = original_seq == traj.final_sequence
            report["parity"].append({"case": case.case_id, "seed": seed, "match": match})
            ok = ok and match
            # Gate 2: same-policy replay parity
            new = adapter.replay_actions(cap, traj, temperature, with_grad=False)
            old_lp = torch.tensor([e.old_logprob for e in traj.events])
            diff = (new["action_logprobs"].cpu() - old_lp).abs()
            entry = {
                "case": case.case_id,
                "seed": seed,
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
            }
            report["replay"].append(entry)
            ok = ok and entry["max_abs_diff"] <= 1e-5
            # FR hard check
            ok = ok and adapter.fr_unchanged(cap, traj, case.full_sequence)
        # also verify feats["res_type"] written by rollout matches the original
        print(json.dumps(report["parity"][-10:], indent=0))
    max_diff = max(r["max_abs_diff"] for r in report["replay"]) if report["replay"] else None
    mean_diff = (
        sum(r["mean_abs_diff"] for r in report["replay"]) / len(report["replay"])
        if report["replay"]
        else None
    )
    report["verdict"] = {
        "parity_pass": all(r["match"] for r in report["parity"]),
        "replay_max_abs_diff": max_diff,
        "replay_mean_abs_diff": mean_diff,
        "replay_pass": max_diff is not None and max_diff <= 1e-5,
        "ok": ok,
    }
    (out / "gate_parity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["verdict"], indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
