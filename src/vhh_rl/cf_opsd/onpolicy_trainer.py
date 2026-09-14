"""Phase E: tiny on-policy EMA refresh loop (task book §56-§64).

Only run when Gate D passes.  Each outer round: behavior rollout -> endpoint
score -> winner/loser pairs -> CF credit -> query -> target construction ->
finite fitting -> EMA refresh.  All teacher/scorer/target data are detached.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import yaml

from ..native_atom14.adapter import NativeDesignAdapter
from ..native_atom14.checkpoint import load_base_model, make_policy_reference, trainable_score_params
from ..native_atom14.dpo_trainer import load_conditioning, move_conditioning
from ..native_atom14.reward import make_reward_adapter
from ..cli.common import case_spec_for
from .behavior_ema import ema_update
from .credit import pair_credit
from .rollout import collect_case_rollouts
from .target_builder import build_target
from .evaluator import evaluate, load_cases

DEVICE = "cuda"
ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")


def run_onpolicy(base_checkpoint: Path, config_path: Path, query_probe: Path,
                 target_probe: Path, output_dir: Path, *, seed: int = 20260914) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    train_cases = cfg["train_cases"]
    heldout_cases = cfg["heldout_cases"]
    K = int(cfg["rollout"]["K"])
    rounds = int(cfg["outer"]["rounds"])
    decay = float(cfg["outer"]["behavior_ema_decay"])
    fit_updates = int(cfg["fit"]["updates_per_round"])
    variant = cfg["fit"].get("loss", "target_mask_only")
    q_star = float(json.loads(query_probe.read_text())["q_star_progress"])
    gate_b = json.loads(target_probe.read_text())
    radius = float(gate_b["selected_radius"])

    torch.manual_seed(seed)
    base = load_base_model(base_checkpoint, device=DEVICE)
    student, _ref = make_policy_reference(base, device=DEVICE)
    behavior, _ = make_policy_reference(base, device=DEVICE)  # separate copy
    params = trainable_score_params(student)
    optimizer = torch.optim.AdamW(params, lr=float(cfg["optimizer"]["lr"]), weight_decay=0.0)
    adapter = NativeDesignAdapter(device=0, sampling_steps=int(cfg["rollout"]["sampling_steps"]),
                                  diffusion_batch_size=K)
    cases = load_cases(train_cases)
    scorer = make_reward_adapter(cache_path=ROOT / "runs/cf_opsd/reward_cache.sqlite")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditionings = {cid: move_conditioning(load_conditioning(
        ROOT / "runs/native_pool/conditioning", cid)) for cid in train_cases}

    history = []
    start = time.time()
    for outer in range(1, rounds + 1):
        round_targets = []
        for cid in train_cases:
            case = cases[cid]
            seed_r = seed * 100 + outer
            trajectories, _ = collect_case_rollouts(
                adapter, case.structure_path.parent / "design.yaml", cid,
                output_dir / f"rollout_r{outer}" / cid, num_designs=K, seed=seed_r)
            seqs = [t.endpoint_sequence for t in trajectories if not t.contains_invalid]
            if seqs:
                batch = scorer.score_sequences(seqs, spec=case_spec_for(case))
                it = iter(batch.raw_scores.tolist())
                for t in trajectories:
                    if not t.contains_invalid:
                        t.endpoint_reward = float(next(it))
            ranked = sorted([t for t in trajectories if t.endpoint_reward is not None],
                            key=lambda t: t.endpoint_reward, reverse=True)
            if len(ranked) < 2:
                continue
            winner, loser = ranked[0], ranked[-1]
            credit = pair_credit(scorer, case, f"{cid}_w", f"{cid}_l",
                                 winner.endpoint_sequence, loser.endpoint_sequence,
                                 case.design_positions, case.fr_positions)
            n_steps = len(loser.sigmas)
            step = max(0, min(n_steps - 1, int(round(q_star * n_steps)) - 1))
            anchor = loser.anchors[step]
            bt = build_target(anchor, winner.endpoint_coords, loser.feats_common,
                              sorted(credit["credits"]), credit["credits"], radius)
            if not bt.touched.any():
                continue
            round_targets.append({"case_id": cid, "anchor": anchor, "target": bt.target_coords,
                                  "touched": bt.touched, "credits": credit["credits"],
                                  "query": loser.query_states[step], "sigma": loser.sigmas[step],
                                  "behavior_endpoint": loser.endpoint_coords,
                                  "behavior_reward": loser.endpoint_reward})

        # finite fitting
        for step_i in range(1, fit_updates + 1):
            rec = round_targets[step_i % len(round_targets)]
            cond = conditionings[rec["case_id"]]
            kwargs = {"s_inputs": cond["s_inputs"], "s_trunk": cond["s_trunk"],
                      "feats": cond["feats"], "multiplicity": 1,
                      "diffusion_conditioning": cond["diffusion_conditioning"]}
            q = rec["query"].to(DEVICE).float().unsqueeze(0)
            sigma = torch.tensor([float(rec["sigma"])], device=DEVICE)
            target = rec["target"].to(DEVICE).float().unsqueeze(0)
            feats = cond["feats"]
            pad = feats["atom_pad_mask"].reshape(-1).bool()
            fake = feats["fake_atom_mask"].reshape(-1).bool()
            token_of_atom = feats["atom_to_token"]
            if token_of_atom.dim() == 3:
                token_of_atom = token_of_atom.squeeze(0)
            token_of_atom = token_of_atom.int().argmax(-1)
            mask = torch.zeros(pad.shape[0], dtype=torch.bool, device=DEVICE)
            for p in rec["credits"]:
                mask |= (token_of_atom == int(p))
            mask &= fake & pad
            optimizer.zero_grad(set_to_none=True)
            denoised, _ = student.structure_module.preconditioned_network_forward(
                q, sigma, training=False, network_condition_kwargs=kwargs)
            loss = ((denoised.float() - target) ** 2)[:, mask, :].sum(dim=-1).mean()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        ema_update(behavior.structure_module, student.structure_module, decay)

        heldout = evaluate(heldout_cases, None, tag=f"onpolicy_r{outer}_behavior",
                           run_root=output_dir / f"eval_r{outer}_behavior")
        row = {"outer_round": outer, "n_targets": len(round_targets),
               "behavior_train_reward_mean": sum(r["behavior_reward"] for r in round_targets)
               / max(1, len(round_targets)),
               "heldout_reward_mean": heldout["reward_mean"],
               "scorer_queries_cumulative": None,
               "optimizer_updates_cumulative": outer * fit_updates,
               "seconds": time.time() - start}
        history.append(row)
        with (output_dir / "onpolicy_metrics.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[onpolicy] round {outer} heldout={row['heldout_reward_mean']}", flush=True)
    scorer.close()
    return {"rounds": rounds, "history": history}
