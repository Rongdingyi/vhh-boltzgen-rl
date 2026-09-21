#!/usr/bin/env python
"""Gate 1: capture exact sampler states for K-sibling branching (§23)."""
from __future__ import annotations

import argparse
import json

import torch

import _common as C  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=C.TRAIN_CASES)
    parser.add_argument("--source-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    steps = {C.progress_to_step(p) for p in C.CANDIDATE_PROGRESS}
    out_dir = C.GATE1_DIR / "prefix"
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = C.load_cases(args.cases)
    adapter = C.make_adapter(C.BASE_CKPT, num_designs=args.source_samples)
    from vhh_rl.cf_opsd.rollout import collect_case_rollouts

    manifest = {}
    for case_id in args.cases:
        case = cases[case_id]
        trajectories, info = collect_case_rollouts(
            adapter, C.spec_path(case), case_id, C.RUN_ROOT / "gate1" / "rollout" / case_id,
            num_designs=args.source_samples, seed=case.seed_base + args.seed,
            cond_steps=steps, state_steps=steps)
        payload = {
            "case_id": case_id,
            "sampler_states": info["sampler_states"],
            "contexts": {int(k): v for k, v in info["contexts"].items()},
            "cond_kwargs_steps": {int(k): v for k, v in info["cond_kwargs_steps"].items()},
            "matches": info["matches"],
            "multiplicity": info["multiplicity"],
            "sampling_scales": info.get("sampling_scales", {}),
            "n_net_calls": info["n_net_calls"],
            "seed": case.seed_base + args.seed,
        }
        path = out_dir / f"{case_id}.pt"
        torch.save(payload, path)
        manifest[case_id] = {
            "prefix": str(path), "multiplicity": info["multiplicity"],
            "state_steps": sorted(steps), "matches": info["matches"],
            "sampling_scales": info.get("sampling_scales", {}),
            "endpoint_sequences": [t.endpoint_sequence for t in trajectories],
        }
        print(f"[gate1-prefix] {case_id}: steps={sorted(steps)} "
              f"multiplicity={info['multiplicity']}", flush=True)
    C.write_json(C.GATE1_DIR / "prefix_manifest.json", manifest)


if __name__ == "__main__":
    main()
