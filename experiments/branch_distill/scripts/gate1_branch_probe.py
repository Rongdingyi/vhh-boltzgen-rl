#!/usr/bin/env python
"""Gate 1: K-sibling branching probe + group/progress gates (§21-§27)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys

import torch

import _common as C  # noqa: E402
from vhh_rl.branch_distill import gates  # noqa: E402
from vhh_rl.branch_distill.branch_rollout import run_branch_group, slice_conditioning
from vhh_rl.branch_distill.metrics import group_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--siblings", type=int, default=C.K_SIBLINGS)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--progress", nargs="*", type=float,
                        default=C.CANDIDATE_PROGRESS)
    args = parser.parse_args()
    from vhh_rl.native_atom14.checkpoint import load_base_model

    cases = C.load_cases(C.TRAIN_CASES)
    model = load_base_model(C.BASE_CKPT, device=C.DEVICE)
    model.eval()
    diffusion = model.structure_module
    scorer = C.make_scorer()

    groups = []
    rows = []
    per_progress: dict[float, list[dict]] = {p: [] for p in args.progress}
    for case_id in C.TRAIN_CASES:
        case = cases[case_id]
        payload = torch.load(C.GATE1_DIR / "prefix" / f"{case_id}.pt",
                             map_location="cpu", weights_only=False)
        scales = payload.get("sampling_scales", {}) or {}
        n_sources = len(payload["matches"])
        for source_index in range(min(n_sources, 2)):
            for progress in args.progress:
                step = C.progress_to_step(progress)
                if step not in payload["sampler_states"]:
                    continue
                batch_index = payload["matches"][source_index]
                pre_state = payload["sampler_states"][step][batch_index]
                cond = payload["cond_kwargs_steps"][step]
                cond = slice_conditioning(cond, batch_index)
                tag = f"{case_id}:{source_index}:{step}".encode()
                group_seed = (int(hashlib.sha1(tag).hexdigest()[:6], 16) % 100000) + args.seed
                group = run_branch_group(
                    diffusion=diffusion, case_id=case_id,
                    source_sample_index=source_index, progress=progress,
                    start_step=step, pre_state=pre_state,
                    conditioning=cond, multiplicity=args.siblings,
                    num_sampling_steps=C.SAMPLING_STEPS, group_seed=group_seed,
                    reference_sequence=case.full_sequence,
                    fr_positions=case.fr_positions,
                    design_positions=case.design_positions,
                    scorer=scorer, spec=C.case_spec(case),
                    step_scale=scales.get("step_scale"),
                    noise_scale=scales.get("noise_scale"))
                metrics = group_metrics(group.siblings, case.design_positions)
                group.meta["group_metrics"] = metrics
                groups.append(group)
                per_progress[progress].append(metrics)
                rows.append({"case_id": case_id, "source_index": source_index,
                             "progress": progress, "start_step": step,
                             "group_seed": group_seed, **metrics})
                print(f"[gate1] {case_id} src{source_index} {progress:.2f}: "
                      f"valid={metrics['valid_rate']:.2f} "
                      f"unique={metrics['n_unique_valid_sequences']} "
                      f"std={metrics['reward_std']:.3f} "
                      f"bmm={metrics['best_minus_median']}", flush=True)
    scorer.close()

    torch.save({"groups": [g.__dict__ for g in groups]},
               C.GATE1_DIR / "branch_groups.pt")
    fields = sorted({k for r in rows for k in r})
    with (C.GATE1_DIR / "group_metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    progress_verdicts = {str(p): gates.gate1_progress_pass(per_progress[p])
                         for p in args.progress}
    selected = None
    for progress in sorted(args.progress):
        if progress_verdicts[str(progress)]["pass"]:
            selected = progress
            break
    payload = {
        "protocol_sha256": C.protocol_hash(),
        "n_groups": len(groups),
        "progress": progress_verdicts,
        "selected_progress": selected,
        "pass": selected is not None,
    }
    C.write_json(C.GATE1_DIR / "gate1.json", payload)
    if selected is not None:
        C.write_json(C.GATE1_DIR / "selected_progress.json",
                     {"selected_progress": selected,
                      "start_step": C.progress_to_step(selected)})
    print(json.dumps({"gate1_pass": payload["pass"], "selected": selected}, indent=1))
    if selected is None:
        sys.exit(1)   # §26/§66: STOP on Gate 1 FAIL


if __name__ == "__main__":
    main()
