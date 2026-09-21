#!/usr/bin/env python
"""Phase 0 training: A offline / B online all-changed / V verified control.

Task book §12: three train seeds, 4 rounds x 25 updates, EMA 0.99, branch
progress 0.60, K=8.  V reuses B's exact pair banks and update schedules
(§7); only ``changed_positions_verified`` is attached, via the script-level
geometry path (the main runtime path stays geometry-free by design).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

import _common as C  # noqa: E402


def _group_seed(case_id: str, seed: int, round_index: int) -> int:
    tag = f"{case_id}:{seed}:{round_index}".encode()
    return int(hashlib.sha1(tag).hexdigest()[:6], 16) % 100000 + seed


def build_groups_factory(cases, scorer, seed: int):
    """Fresh behavior rollouts -> captured state -> K siblings (§12.1)."""
    from vhh_rl.branch_distill.branch_rollout import run_branch_group, slice_conditioning
    from vhh_rl.cf_opsd.rollout import collect_case_rollouts
    from vhh_rl.native_atom14.adapter import NativeDesignAdapter
    from vhh_rl.native_atom14.checkpoint import load_base_model

    step = C.progress_to_step(C.BRANCH_PROGRESS)

    def build_groups(round_index: int, behavior_ckpt, cfg):
        groups = []
        model = load_base_model(behavior_ckpt, device=C.DEVICE)
        model.eval()
        for case_id in C.TRAIN_CASES:
            case = cases[case_id]
            adapter = NativeDesignAdapter(design_ckpt=Path(behavior_ckpt),
                                          sampling_steps=C.SAMPLING_STEPS,
                                          diffusion_batch_size=C.K_SIBLINGS)
            adapter.num_designs = 1
            roll_dir = Path(cfg.output_dir) / f"rollout_r{round_index}" / case_id
            grad_was = torch.is_grad_enabled()
            try:
                _traj, info = collect_case_rollouts(
                    adapter, C.spec_path(case), case_id, roll_dir, num_designs=1,
                    seed=case.seed_base + seed + 1000 * round_index,
                    cond_steps={step}, state_steps={step})
            finally:
                torch.set_grad_enabled(grad_was)
            scales = info.get("sampling_scales", {}) or {}
            cond = slice_conditioning(info["cond_kwargs_steps"][step], 0)
            pre_state = info["sampler_states"][step][info["matches"][0]]
            group = run_branch_group(
                diffusion=model.structure_module, case_id=case_id,
                source_sample_index=0, progress=C.BRANCH_PROGRESS, start_step=step,
                pre_state=pre_state, conditioning=cond,
                multiplicity=C.K_SIBLINGS, num_sampling_steps=C.SAMPLING_STEPS,
                group_seed=_group_seed(case_id, seed, round_index),
                reference_sequence=case.full_sequence,
                fr_positions=case.fr_positions,
                design_positions=case.design_positions,
                scorer=scorer, spec=C.case_spec(case),
                step_scale=scales.get("step_scale"),
                noise_scale=scales.get("noise_scale"))
            groups.append(group)
        return groups

    return build_groups


def attach_verified(pairs, cfg) -> None:
    """Phase 0 control only: endpoint-carrier verified support (§7)."""
    from vhh_rl.branch_distill.local_target import (carrier_audit,
                                                    carrier_positions_match,
                                                    verified_changed_positions)

    cases = C.load_cases(C.TRAIN_CASES)
    for pair in pairs:
        case = cases[pair.case_id]
        feats = pair.conditioning["feats"]
        # carrier_audit(peer_endpoint, teacher_endpoint, ...): the winner is the
        # teacher, the loser is the peer whose anchor receives the geometry
        carrier = carrier_audit(pair.loser_coords, pair.winner_coords, feats,
                                pair.changed_positions_all, case.full_sequence,
                                case.fr_positions)
        matches = carrier_positions_match(carrier["sequence"], pair.winner_sequence,
                                          pair.changed_positions_all)
        verified = verified_changed_positions(matches, pair.changed_positions_all)
        if verified:
            pair.changed_positions_verified = tuple(verified)
        else:
            # no carrier-matched position: the update is logged and skipped by
            # the trainer, and the skipped count is reported (V only)
            pair.changed_positions_verified = None
            pair.meta["verified_empty"] = True
        pair.meta["carrier_original_rate"] = (len(verified) / len(matches)
                                              if matches else 0.0)


def arm_offline(seed: int) -> dict:
    """Arm A: offline fixed diff_only pairs, all changed positions (§12.2A)."""
    from vhh_rl.native_atom14.weighted_dpo import run_weighted_dpo

    out_dir = C.seed_dir(seed) / "A"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_all = [json.loads(l) for l in
                 (C.ROOT / "runs/native_pool/pairs_train.jsonl").open()]
    keep = set(C.TRAIN_CASES)
    pairs = [p for p in pairs_all if p["case_id"] in keep]
    weights_all = json.loads((C.ROOT / "runs/paper_stage/weights/"
                              "ablation_weights.json").read_text())
    weights = {"pairs": {k: v for k, v in weights_all["pairs"].items()
                         if v["case_id"] in keep}}
    pairs_path = out_dir / "pairs_offline.jsonl"
    weights_path = out_dir / "weights_offline.json"
    pairs_path.write_text("".join(json.dumps(p) + "\n" for p in pairs))
    weights_path.write_text(json.dumps(weights, indent=1))
    summary = run_weighted_dpo(
        base_checkpoint=C.BASE_CKPT, pairs_path=pairs_path,
        pool_root=C.ROOT / "runs/native_pool",
        conditioning_dir=C.ROOT / "runs/native_pool/conditioning",
        weights_path=weights_path, output_dir=out_dir / "run", variant="diff_only",
        beta=10.0, lr=1e-5, max_steps=100, checkpoint_every=100, seed=seed,
        log_tag=f"online-pref-A-{seed}")
    final = out_dir / "run" / "checkpoint_0100.pt"
    (out_dir / "student_r4.pt").write_bytes(final.read_bytes())
    C.write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["a", "b", "v"])
    parser.add_argument("--seed", type=int, required=True,
                        choices=C.TRAIN_SEEDS)
    args = parser.parse_args()
    from vhh_rl.online_pref.trainer import OnlineArmConfig, run_online_arm

    if args.arm == "a":
        summary = arm_offline(args.seed)
        print(json.dumps({"arm": "A", "seed": args.seed,
                          "final_loss": summary.get("final_loss")}, indent=1))
        return
    cases = C.load_cases(C.TRAIN_CASES)
    scorer = C.make_scorer()
    design_positions = {cid: cases[cid].design_positions for cid in C.TRAIN_CASES}
    bank_root = C.seed_dir(args.seed) / "B"
    deps = {
        "base_checkpoint": C.BASE_CKPT,
        "design_positions": design_positions,
        "scorer": scorer,
        "attach_verified": attach_verified,
    }
    if args.arm == "b":
        deps["build_groups"] = build_groups_factory(cases, scorer, args.seed)
        cfg = OnlineArmConfig(arm="B", seed=args.seed, support="all",
                              data_source="fresh", output_dir=str(bank_root),
                              bank_root=str(bank_root))
    else:
        cfg = OnlineArmConfig(arm="V", seed=args.seed, support="verified",
                              data_source="bank",
                              bank_source_root=str(bank_root),
                              output_dir=str(C.seed_dir(args.seed) / "V"),
                              bank_root=str(C.seed_dir(args.seed) / "V"))
    summary = run_online_arm(cfg, deps)
    scorer.close()
    print(json.dumps({"arm": cfg.arm, "seed": args.seed,
                      "reward_queries": summary["reward_queries_total"],
                      "pairs": summary["pair_records_total"]}, indent=1))


if __name__ == "__main__":
    main()
