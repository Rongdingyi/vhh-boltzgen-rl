#!/usr/bin/env python
"""Gate 3: train one arm (A offline / B online DPO / C,D sibling distill).

Task book §41-§52: per-round behavior refresh, common round-1 branch bank,
per-update and per-round logging, EMA 0.99, final student_r4 checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

torch.set_float32_matmul_precision("high")  # same-query replay requires the sampler precision

import _common as C  # noqa: E402
from vhh_rl.branch_distill.behavior_loop import (
    Gate3Config, build_online_records, clear_round_dir, run_online_update,
)
from vhh_rl.branch_distill.branch_rollout import run_branch_group, slice_conditioning
from vhh_rl.branch_distill.query_fit import network_kwargs


def _load_gate3_config() -> Gate3Config:
    frozen = json.loads((C.GATE1_DIR / "selected_progress.json").read_text())
    cfg = Gate3Config(train_cases=C.TRAIN_CASES,
                      selected_progress=frozen["selected_progress"])
    return cfg


def arm_a(cfg: Gate3Config, out_dir: Path, seed: int) -> dict:
    """Offline diff_only DPO baseline on the fixed 4 train cases (§41/§49)."""
    from vhh_rl.native_atom14.weighted_dpo import run_weighted_dpo
    pairs_all = [json.loads(l) for l in (C.ROOT / "runs/native_pool/pairs_train.jsonl").open()]
    keep = set(C.TRAIN_CASES)
    pairs = [p for p in pairs_all if p["case_id"] in keep]
    weights_all = json.loads((C.ROOT / "runs/paper_stage/weights/ablation_weights.json").read_text())
    weights = {"pairs": {k: v for k, v in weights_all["pairs"].items()
                         if v["case_id"] in keep}}
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "pairs_offline.jsonl"
    weights_path = out_dir / "weights_offline.json"
    pairs_path.write_text("".join(json.dumps(p) + "\n" for p in pairs))
    weights_path.write_text(json.dumps(weights, indent=1))
    summary = run_weighted_dpo(
        base_checkpoint=C.BASE_CKPT, pairs_path=pairs_path, pool_root=C.ROOT / "runs/native_pool",
        conditioning_dir=C.ROOT / "runs/native_pool/conditioning",
        weights_path=weights_path, output_dir=out_dir / "run", variant="diff_only",
        beta=10.0, lr=cfg.lr, max_steps=cfg.total_updates,
        checkpoint_every=cfg.total_updates, seed=seed, log_tag="branch-gate3-a")
    final = out_dir / "run" / f"checkpoint_{cfg.total_updates:04d}.pt"
    target = out_dir / "student_r4.pt"
    target.write_bytes(final.read_bytes())
    return summary


def _build_round_groups(cfg: Gate3Config, round_idx: int, behavior_ckpt: Path,
                        scorer, cases, seed: int) -> list:
    """Behavior rollouts -> captured states -> K siblings (round 1 is shared)."""
    from vhh_rl.cf_opsd.rollout import collect_case_rollouts
    from vhh_rl.native_atom14.checkpoint import load_base_model
    groups = []
    model = load_base_model(behavior_ckpt, device=C.DEVICE)
    model.eval()
    step = C.progress_to_step(cfg.selected_progress)
    for case_id in cfg.train_cases:
        case = cases[case_id]
        cache = C.COMMON_ROUND1 / f"{case_id}.pt"
        roll_dir = C.GATE3_DIR / f"rollout_r{round_idx}" / case_id
        if round_idx == 1 and cache.is_file():
            payload = torch.load(cache, map_location="cpu", weights_only=False)
        else:
            adapter = C.make_adapter(behavior_ckpt, num_designs=1)
            _traj, info = collect_case_rollouts(
                adapter, C.spec_path(case), case_id, roll_dir, num_designs=1,
                seed=case.seed_base + seed, cond_steps={step}, state_steps={step})
            payload = {"case_id": case_id,
                       "sampler_states": info["sampler_states"],
                       "cond_kwargs_steps": info["cond_kwargs_steps"],
                       "matches": info["matches"],
                       "multiplicity": info["multiplicity"],
                       "sampling_scales": info.get("sampling_scales", {})}
            if round_idx == 1:
                C.COMMON_ROUND1.mkdir(parents=True, exist_ok=True)
                torch.save(payload, cache)
        cond = slice_conditioning(payload["cond_kwargs_steps"][step], 0)
        pre_state = payload["sampler_states"][step][payload["matches"][0]]
        group_seed = (int(hashlib.sha1(
            f"{case_id}:{round_idx}:{seed}".encode()).hexdigest()[:6], 16) % 100000) + seed
        group = run_branch_group(
            diffusion=model.structure_module, case_id=case_id,
            source_sample_index=0, progress=cfg.selected_progress,
            start_step=step, pre_state=pre_state, conditioning=cond,
            multiplicity=cfg.siblings_per_case, num_sampling_steps=cfg.sampling_steps,
            group_seed=group_seed, reference_sequence=case.full_sequence,
            fr_positions=case.fr_positions, design_positions=case.design_positions,
            scorer=scorer, spec=C.case_spec(case),
            step_scale=(payload.get("sampling_scales") or {}).get("step_scale"),
            noise_scale=(payload.get("sampling_scales") or {}).get("noise_scale"))
        groups.append(group)
    return groups


def online_arm(arm: str, cfg: Gate3Config, out_dir: Path, seed: int) -> dict:
    from vhh_rl.native_atom14.checkpoint import (
        load_base_model, make_policy_reference, parameter_drift,
        save_native_checkpoint, trainable_score_params,
    )
    from vhh_rl.branch_distill.behavior_loop import ema_refresh

    cases = C.load_cases(cfg.train_cases)
    scorer = C.make_scorer()
    clear_round_dir(out_dir)
    base = load_base_model(C.BASE_CKPT, device=C.DEVICE)
    student, reference = make_policy_reference(base, device=C.DEVICE)
    behavior = load_base_model(C.BASE_CKPT, device=C.DEVICE)
    params = trainable_score_params(student)
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = random.Random(seed)
    step = 0
    metrics_path = out_dir / "train_metrics.jsonl"

    def log_row(row: dict) -> None:
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    rounds_log = []
    for round_idx in range(1, cfg.rounds + 1):
        behavior_ckpt = (C.BASE_CKPT if round_idx == 1
                         else out_dir / f"behavior_r{round_idx - 1}.pt")
        groups = _build_round_groups(cfg, round_idx, behavior_ckpt, scorer, cases, seed)
        records, audits = [], []
        for case_id in cfg.train_cases:
            case = cases[case_id]
            sub = [g for g in groups if g.case_id == case_id]
            recs, auds = build_online_records(sub, cfg, case.design_positions,
                                              case.full_sequence, case.fr_positions)
            records += recs
            audits += auds
        if not records:
            raise SystemExit(f"arm {arm}: no eligible online records in round {round_idx}")
        for _ in range(cfg.updates_per_round):
            step += 1
            run_online_update(arm, student, reference, records, cfg,
                              rng, step, optimizer, params, log_row)
        ema_refresh(behavior, student, cfg.ema_decay)
        save_native_checkpoint(C.BASE_CKPT, student,
                               out_dir / f"student_r{round_idx}.pt",
                               {"arm": arm, "round": round_idx, "step": step})
        save_native_checkpoint(C.BASE_CKPT, behavior,
                               out_dir / f"behavior_r{round_idx}.pt",
                               {"arm": arm, "round": round_idx})
        rounds_log.append({
            "round": round_idx, "updates": cfg.updates_per_round,
            "queries_cumulative": len(records),
            "n_valid_siblings": sum(len([s for s in g.siblings
                                         if s.reward is not None]) for g in groups),
            "param_drift": parameter_drift(student, reference)["total"],
        })
        print(f"[gate3-{arm}] round {round_idx}: records={len(records)} "
              f"step={step}", flush=True)
    scorer.close()
    C.write_json(out_dir / "rounds.json", rounds_log)
    final = out_dir / f"student_r{cfg.rounds}.pt"
    (out_dir / "student_r4.pt").write_bytes(final.read_bytes())
    return {"arm": arm, "rounds": rounds_log, "updates": step}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["a", "b", "c", "d"])
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--override-gate", default=None)
    args = parser.parse_args()
    C.require_gate(C.GATE2_DIR / "gate2.json", "pass", True, args.override_gate, "Gate 2")
    cfg = _load_gate3_config()
    out_dir = C.GATE3_DIR / args.arm
    if args.arm == "a":
        summary = arm_a(cfg, out_dir, args.seed)
    else:
        summary = online_arm(args.arm.upper(), cfg, out_dir, args.seed)
    C.write_json(out_dir / "train_summary.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
