"""Phase E: tiny on-policy EMA refresh loop (task book §56-§64).

Only run when Gate D passes.  Each outer round:
  behavior rollout -> endpoint score -> winner/loser -> CF credit -> query ->
  target -> finite student fitting -> EMA refresh -> behavior checkpoint ->
  next round's rollout uses the EMA'd behavior.

Wiring guarantees (audit fixes):
  * ``behavior`` is an independent deep copy of the base model, never the
    student object itself;
  * every round's rollouts are produced by the *current behavior checkpoint*
    (``adapter.design_ckpt`` is switched to the EMA checkpoint);
  * held-out evaluation uses the *student checkpoint* saved that round, not
    the native base.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import torch
import yaml

from ..native_atom14.adapter import NativeDesignAdapter, _sha256_file
from ..native_atom14.checkpoint import (
    load_base_model, make_policy_reference, save_native_checkpoint,
    trainable_score_params,
)
from ..native_atom14.dpo_trainer import load_conditioning, move_conditioning
from ..native_atom14.reward import make_reward_adapter
from ..cli.common import case_spec_for
from .behavior_ema import ema_update
from .credit import pair_credit
from .rollout import collect_case_rollouts
from .target_builder import build_target

DEVICE = "cuda"
ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
REQUIRED_KEYS = ("train_cases", "heldout_cases", "rollout", "outer", "fit", "optimizer")


def _require(cfg: dict, keys=REQUIRED_KEYS) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise KeyError(f"on-policy config missing required sections: {missing}")


def set_adapter_checkpoint(adapter, ckpt: Path) -> None:
    """Switch the adapter (and its recorded sha) to a new design checkpoint."""
    ckpt = Path(ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    adapter.design_ckpt = ckpt
    adapter.checkpoint_sha256 = _sha256_file(ckpt)


def default_deps() -> dict:
    return {
        "adapter_factory": lambda K, steps: NativeDesignAdapter(
            device=0, sampling_steps=steps, diffusion_batch_size=K),
        "rollout_fn": collect_case_rollouts,
        "evaluate_fn": None,  # resolved lazily to cf_opsd.evaluator.evaluate
        "scorer_factory": lambda: make_reward_adapter(
            cache_path=ROOT / "runs/cf_opsd/reward_cache.sqlite"),
        "credit_fn": pair_credit,
        "target_fn": build_target,
        "load_base": load_base_model,
        "make_policy_ref": make_policy_reference,
        "save_ckpt": save_native_checkpoint,
    }


def run_onpolicy(base_checkpoint: Path, case_config: Path, loop_config: Path,
                 query_probe: Path, target_probe: Path, output_dir: Path, *,
                 seed: int = 20260914, deps: dict | None = None) -> dict:
    cfg = {**yaml.safe_load(Path(case_config).read_text()),
           **yaml.safe_load(Path(loop_config).read_text())}
    _require(cfg)
    train_cases = list(cfg["train_cases"])
    heldout_cases = list(cfg["heldout_cases"])
    K = int(cfg["rollout"]["K"])
    steps = int(cfg["rollout"]["sampling_steps"])
    rounds = int(cfg["outer"]["rounds"])
    decay = float(cfg["outer"]["behavior_ema_decay"])
    fit_updates = int(cfg["fit"]["updates_per_round"])
    variant = cfg["fit"].get("loss", "target_mask_only")
    lr = float(cfg["optimizer"]["lr"])
    max_grad_norm = float(cfg["optimizer"].get("max_grad_norm", 1.0))
    q_star = float(json.loads(Path(query_probe).read_text())["q_star_progress"])
    gate_b = json.loads(Path(target_probe).read_text())
    radius = float(gate_b["selected_radius"])

    deps = {**default_deps(), **(deps or {})}
    if deps["evaluate_fn"] is None:
        from .evaluator import evaluate as _evaluate
        deps["evaluate_fn"] = _evaluate
    if "load_cases" not in deps:
        from .evaluator import load_cases as _load_cases
        deps["load_cases"] = _load_cases

    torch.manual_seed(seed)
    base = deps["load_base"](base_checkpoint, device=DEVICE)
    student, _ref = deps["make_policy_ref"](base, device=DEVICE)
    behavior = copy.deepcopy(base)          # independent copy (audit fix)
    behavior.eval()
    for p in behavior.parameters():
        p.requires_grad_(False)
    params = trainable_score_params(student)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=float(
        cfg["optimizer"].get("weight_decay", 0.0)))

    adapter = deps["adapter_factory"](K, steps)
    set_adapter_checkpoint(adapter, Path(base_checkpoint))  # round 1: base behavior
    cases = deps["load_cases"](train_cases)
    scorer = deps["scorer_factory"]()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditionings = {cid: move_conditioning(load_conditioning(
        ROOT / "runs/native_pool/conditioning", cid)) for cid in train_cases}

    history = []
    queries_cumulative = 0
    updates_cumulative = 0
    start = time.time()
    for outer in range(1, rounds + 1):
        round_targets = []
        for cid in train_cases:
            case = cases[cid]
            seed_r = seed * 100 + outer
            trajectories, info = deps["rollout_fn"](
                adapter, case.structure_path.parent / "design.yaml", cid,
                output_dir / f"rollout_r{outer}" / cid, num_designs=K, seed=seed_r)
            seqs, slots = [], []
            for t in trajectories:
                if not t.contains_invalid and t.fr_mismatch == 0:
                    seqs.append(t.endpoint_sequence)
                    slots.append(t)
            if seqs:
                batch = scorer.score_sequences(seqs, spec=case_spec_for(case))
                for t, v in zip(slots, batch.raw_scores.tolist()):
                    t.endpoint_reward = float(v)
                queries_cumulative += len(seqs)
            ranked = sorted([t for t in trajectories if t.endpoint_reward is not None],
                            key=lambda t: t.endpoint_reward, reverse=True)
            if len(ranked) < 2:
                continue
            winner, loser = ranked[0], ranked[-1]
            credit = deps["credit_fn"](scorer, case, f"{cid}_w", f"{cid}_l",
                                       winner.endpoint_sequence, loser.endpoint_sequence,
                                       case.design_positions, case.fr_positions)
            queries_cumulative += credit["n_queries"]
            n_steps = len(loser.sigmas)
            step = max(0, min(n_steps - 1, int(round(q_star * n_steps)) - 1))
            anchor = loser.anchors[step]
            bt = deps["target_fn"](anchor, winner.endpoint_coords, loser.feats_common,
                                   sorted(credit["credits"]), credit["credits"], radius)
            if not bt.touched.any():
                continue
            round_targets.append({"case_id": cid, "anchor": anchor,
                                  "target": bt.target_coords, "touched": bt.touched,
                                  "credits": credit["credits"],
                                  "query": loser.query_states[step],
                                  "sigma": loser.sigmas[step],
                                  "behavior_endpoint": loser.endpoint_coords,
                                  "behavior_reward": winner.endpoint_reward})
        if not round_targets:
            print(f"[onpolicy] round {outer}: no targets, skipped fitting", flush=True)
        for step_i in range(1, fit_updates + 1):
            if not round_targets:
                break
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
            grad = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            updates_cumulative += 1

        # EMA refresh -> behavior checkpoint used by the NEXT round's rollouts
        ema_update(behavior.structure_module, student.structure_module, decay)
        behavior_ckpt = output_dir / f"behavior_r{outer}.pt"
        deps["save_ckpt"](base_checkpoint, behavior, behavior_ckpt,
                          {"method": "cf_opsd_onpolicy", "role": "behavior",
                           "outer_round": outer, "seed": seed})
        set_adapter_checkpoint(adapter, behavior_ckpt)

        # held-out: evaluate the STUDENT checkpoint of this round (audit fix)
        student_ckpt = output_dir / f"student_r{outer}.pt"
        deps["save_ckpt"](base_checkpoint, student, student_ckpt,
                          {"method": "cf_opsd_onpolicy", "role": "student",
                           "outer_round": outer, "seed": seed})
        heldout = deps["evaluate_fn"](heldout_cases, student_ckpt,
                                      tag=f"onpolicy_r{outer}_student",
                                      run_root=output_dir / f"eval_r{outer}_student")
        b_params = dict(behavior.named_parameters())
        drift_sq = 0.0
        for n, p in student.named_parameters():
            if n in b_params:
                drift_sq += float(((p.detach() - b_params[n].detach()).float() ** 2).sum())
        drift = drift_sq ** 0.5
        row = {
            "outer_round": outer,
            "n_targets": len(round_targets),
            "behavior_train_reward_mean": (sum(r["behavior_reward"] for r in round_targets)
                                           / max(1, len(round_targets))),
            "heldout_reward_mean_student": heldout["reward_mean"],
            "student_behavior_param_distance": drift,
            "scorer_queries_cumulative": queries_cumulative,
            "optimizer_updates_cumulative": updates_cumulative,
            "behavior_checkpoint": str(behavior_ckpt),
            "student_checkpoint": str(student_ckpt),
            "seconds": time.time() - start,
        }
        history.append(row)
        with (output_dir / "onpolicy_metrics.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[onpolicy] round {outer} heldout(student)={row['heldout_reward_mean_student']}",
              flush=True)
    scorer.close()
    return {"rounds": rounds, "history": history}
