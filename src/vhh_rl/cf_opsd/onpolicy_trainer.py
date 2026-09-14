"""Phase E: tiny on-policy EMA refresh loop (task book §56-§64).

Only runs when Gate B, Gate C and Gate D all pass; the gate artifacts are
required arguments and are hard-checked before any model work.

Each outer round:
  behavior rollout (pairs_per_case pairs per case) -> endpoint score ->
  winner/loser -> CF credit -> query -> target -> finite student fitting
  (target_mask_only | target_mask_weak_hold) -> EMA refresh ->
  behavior checkpoint for the next round's rollout -> save student checkpoint
  -> held-out evaluation of that student checkpoint.

Heavy imports (BoltzGen adapter, evaluator, trainer helpers) are resolved
lazily through ``default_deps()`` so the module is importable without the
external BoltzGen checkout (CI runs CPU-only wiring tests).
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import torch
import yaml

from .behavior_ema import ema_update

DEVICE = "cuda"
ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
REQUIRED_KEYS = ("train_cases", "heldout_cases", "rollout", "outer", "fit", "optimizer")


def _require(cfg: dict, keys=REQUIRED_KEYS) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise KeyError(f"on-policy config missing required sections: {missing}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_adapter_checkpoint(adapter, ckpt: Path) -> None:
    """Switch the adapter (and its recorded sha) to a new design checkpoint."""
    ckpt = Path(ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    adapter.design_ckpt = ckpt
    adapter.checkpoint_sha256 = _sha256_file(ckpt)


def _default_load_base(checkpoint, device=DEVICE):
    from ..native_atom14.checkpoint import load_base_model
    return load_base_model(checkpoint, device=device)


def _default_make_policy_ref(base, device=DEVICE):
    from ..native_atom14.checkpoint import make_policy_reference
    return make_policy_reference(base, device=device)


def _default_save_ckpt(base_checkpoint, model, path, metadata):
    from ..native_atom14.checkpoint import save_native_checkpoint
    return save_native_checkpoint(base_checkpoint, model, path, metadata)


def _default_adapter(K, steps):
    from ..native_atom14.adapter import NativeDesignAdapter
    return NativeDesignAdapter(device=0, sampling_steps=steps, diffusion_batch_size=K)


def _default_rollout(adapter, spec, case_id, run_root, num_designs, seed):
    from .rollout import collect_case_rollouts
    return collect_case_rollouts(adapter, spec, case_id, run_root,
                                 num_designs=num_designs, seed=seed)


def _default_evaluate(case_ids, ckpt, tag, run_root=None):
    from .evaluator import evaluate
    return evaluate(case_ids, ckpt, tag, run_root=run_root)


def _default_scorer():
    from ..native_atom14.reward import make_reward_adapter
    return make_reward_adapter(cache_path=ROOT / "runs/cf_opsd/reward_cache.sqlite")


def _default_credit(scorer, case, w_id, l_id, w_seq, l_seq, design, fr):
    from .credit import pair_credit
    return pair_credit(scorer, case, w_id, l_id, w_seq, l_seq, design, fr)


def _default_target(anchor, winner, feats, positions, credits, radius, **kw):
    from .target_builder import build_target
    return build_target(anchor, winner, feats, positions, credits, radius, **kw)


def _default_load_cases(case_ids):
    from .evaluator import load_cases
    return load_cases(case_ids)


def _default_load_cond(cond_dir, cid):
    from ..native_atom14.dpo_trainer import load_conditioning
    return load_conditioning(cond_dir, cid)


def _default_move_cond(payload, device=DEVICE):
    from ..native_atom14.dpo_trainer import move_conditioning
    return move_conditioning(payload, device=device)


def _default_spec_fn(case):
    from ..cli.common import case_spec_for
    return case_spec_for(case)


def _default_trainable_params(model):
    from ..native_atom14.checkpoint import trainable_score_params
    return trainable_score_params(model)


def default_deps() -> dict:
    return {
        "adapter_factory": _default_adapter,
        "rollout_fn": _default_rollout,
        "evaluate_fn": _default_evaluate,
        "scorer_factory": _default_scorer,
        "credit_fn": _default_credit,
        "target_fn": _default_target,
        "load_base": _default_load_base,
        "make_policy_ref": _default_make_policy_ref,
        "save_ckpt": _default_save_ckpt,
        "load_cases": _default_load_cases,
        "load_cond": _default_load_cond,
        "move_cond": _default_move_cond,
        "spec_fn": _default_spec_fn,
        "trainable_params": _default_trainable_params,
    }


def check_gates(gate_b_path: Path, gate_c_path: Path, gate_d_path: Path) -> dict:
    """Hard-check Gate B/C/D artifacts; refuse to run otherwise (audit fix C)."""
    gate_b = json.loads(Path(gate_b_path).read_text())
    gate_c = json.loads(Path(gate_c_path).read_text())
    gate_d = json.loads(Path(gate_d_path).read_text())
    if gate_b.get("selected_radius") is None:
        raise RuntimeError("Gate B not passed (no selected radius) -> refusing Phase E")
    if not any(gate_c.get("gate_c", {}).values()):
        raise RuntimeError("Gate C not passed -> refusing Phase E")
    if not any((v or {}).get("reward_superiority") for v in gate_d.get("gate_d", {}).values()):
        raise RuntimeError("Gate D not passed -> refusing Phase E")
    return {"gate_b": gate_b, "gate_c": gate_c, "gate_d": gate_d}


def run_onpolicy(base_checkpoint: Path, case_config: Path, loop_config: Path,
                 query_probe: Path, gate_b_path: Path, gate_c_path: Path,
                 gate_d_path: Path, output_dir: Path, *,
                 seed: int = 20260914, deps: dict | None = None) -> dict:
    cfg = {**yaml.safe_load(Path(case_config).read_text()),
           **yaml.safe_load(Path(loop_config).read_text())}
    _require(cfg)
    gates = check_gates(gate_b_path, gate_c_path, gate_d_path)
    q_star = float(json.loads(Path(query_probe).read_text())["q_star_progress"])
    radius = float(gates["gate_b"]["selected_radius"])

    train_cases = list(cfg["train_cases"])
    heldout_cases = list(cfg["heldout_cases"])
    K = int(cfg["rollout"]["K"])
    steps = int(cfg["rollout"]["sampling_steps"])
    rounds = int(cfg["outer"]["rounds"])
    decay = float(cfg["outer"]["behavior_ema_decay"])
    fit_updates = int(cfg["fit"]["updates_per_round"])
    variant = cfg["fit"].get("loss", "target_mask_only")
    hold_lambda = float(cfg["fit"].get("hold_lambda", 0.05))
    n_pairs = int(cfg.get("pairing", {}).get("pairs_per_case", 1))
    lr = float(cfg["optimizer"]["lr"])
    max_grad_norm = float(cfg["optimizer"].get("max_grad_norm", 1.0))

    deps = {**default_deps(), **(deps or {})}
    torch.manual_seed(seed)
    base = deps["load_base"](base_checkpoint, device=DEVICE)
    student, _ref = deps["make_policy_ref"](base, device=DEVICE)
    behavior = copy.deepcopy(base)          # independent copy (audit fix)
    behavior.eval()
    for p in behavior.parameters():
        p.requires_grad_(False)
    params = deps["trainable_params"](student)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=float(
        cfg["optimizer"].get("weight_decay", 0.0)))

    adapter = deps["adapter_factory"](K, steps)
    set_adapter_checkpoint(adapter, Path(base_checkpoint))
    cases = deps["load_cases"](train_cases)
    scorer = deps["scorer_factory"]()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditionings = {cid: deps["move_cond"](deps["load_cond"](
        ROOT / "runs/native_pool/conditioning", cid), device=DEVICE)
        for cid in train_cases}

    history = []
    queries_cumulative = 0
    updates_cumulative = 0
    start = time.time()
    for outer in range(1, rounds + 1):
        round_targets = []
        for cid in train_cases:
            case = cases[cid]
            seed_r = seed * 100 + outer
            trajectories, _info = deps["rollout_fn"](
                adapter, case.structure_path.parent / "design.yaml", cid,
                output_dir / f"rollout_r{outer}" / cid, num_designs=K, seed=seed_r)
            seqs, slots = [], []
            for t in trajectories:
                if not t.contains_invalid and t.fr_mismatch == 0:
                    seqs.append(t.endpoint_sequence)
                    slots.append(t)
            if seqs:
                batch = scorer.score_sequences(seqs, spec=deps["spec_fn"](case))
                for t, v in zip(slots, batch.raw_scores.tolist()):
                    t.endpoint_reward = float(v)
                queries_cumulative += len(seqs)
            ranked = sorted([t for t in trajectories if t.endpoint_reward is not None],
                            key=lambda t: t.endpoint_reward, reverse=True)
            n_use = min(n_pairs, len(ranked) // 2)
            if n_use < 1:
                continue
            n_steps = len(ranked[0].sigmas)
            step = max(0, min(n_steps - 1, int(round(q_star * n_steps)) - 1))
            for pi in range(n_use):
                winner, loser = ranked[pi], ranked[-1 - pi]   # rank1-vs-rank8, rank2-vs-rank7
                credit = deps["credit_fn"](scorer, case, f"{cid}_w{pi}", f"{cid}_l{pi}",
                                           winner.endpoint_sequence, loser.endpoint_sequence,
                                           case.design_positions, case.fr_positions)
                queries_cumulative += credit["n_queries"]
                anchor = loser.anchors[step]
                bt = deps["target_fn"](anchor, winner.endpoint_coords, loser.feats_common,
                                       sorted(credit["credits"]), credit["credits"], radius)
                if not bt.touched.any():
                    continue
                round_targets.append({
                    "case_id": cid, "pair_index": pi,
                    "anchor": anchor, "target": bt.target_coords, "touched": bt.touched,
                    "credits": credit["credits"],
                    "query": loser.query_states[step], "sigma": loser.sigmas[step],
                    "behavior_reward": winner.endpoint_reward,
                })
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
            denoised = denoised.float()
            loss = ((denoised - target) ** 2)[:, mask, :].sum(dim=-1).mean()
            if variant == "target_mask_weak_hold":
                resolved = feats["atom_resolved_mask"].reshape(-1).bool()
                hold_mask = (pad & resolved & ~mask).to(DEVICE)
                if hold_mask.any():
                    anchor_t = rec["anchor"].to(DEVICE).float().unsqueeze(0)
                    hold = ((denoised - anchor_t) ** 2)[:, hold_mask, :].sum(dim=-1).mean()
                    loss = loss + hold_lambda * hold
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            updates_cumulative += 1

        ema_update(behavior.structure_module, student.structure_module, decay)
        behavior_ckpt = output_dir / f"behavior_r{outer}.pt"
        deps["save_ckpt"](base_checkpoint, behavior, behavior_ckpt,
                          {"method": "cf_opsd_onpolicy", "role": "behavior",
                           "outer_round": outer, "seed": seed})
        set_adapter_checkpoint(adapter, behavior_ckpt)

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
        row = {
            "outer_round": outer,
            "n_targets": len(round_targets),
            "n_pairs_per_case": n_use,
            "fit_variant": variant,
            "behavior_train_reward_mean": (sum(r["behavior_reward"] for r in round_targets)
                                           / max(1, len(round_targets))),
            "heldout_reward_mean_student": heldout["reward_mean"],
            "student_behavior_param_distance": drift_sq ** 0.5,
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
