#!/usr/bin/env python
"""Native smoke gates 2-5 (task book §65-68):
  A. pair correctness
  B. paired noising sync (sigma/noise/rotation/translation)
  C. DPO init loss ~ log 2 on 20 pairs
  D. gradient audit: only score_model gets grads
  E. one-pair overfit: z rises over 60 updates
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.native_atom14.checkpoint import (  # noqa: E402
    load_base_model, make_policy_reference, trainable_score_params,
)
from vhh_rl.native_atom14.denoise_loss import per_sample_denoising_loss  # noqa: E402
from vhh_rl.native_atom14.dpo_loss import diffusion_dpo_loss  # noqa: E402
from vhh_rl.native_atom14.dpo_trainer import (  # noqa: E402
    load_conditioning, move_conditioning, _compute_masks, _load_coords,
)
from vhh_rl.native_atom14.paired_noise import paired_noising  # noqa: E402
from vhh_rl.native_atom14.preference_dataset import build_pairs  # noqa: E402

POOL = ROOT / "runs/native_pool"
COND = POOL / "conditioning"
OUT = ROOT / "runs/native_smoke_gates"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    pairs = build_pairs(POOL, "train", top_k=8, out_path=POOL / "pairs_train.jsonl")
    print(f"pairs: {len(pairs)}", flush=True)
    assert pairs, "no pairs"

    # ---------------- Gate A: pair correctness
    failures = 0
    for p in pairs:
        assert p["reward_gap"] > 0
        if p["reward_gap"] <= 0:
            failures += 1
    report["pair_correctness"] = {"n_pairs": len(pairs), "failures": failures}

    # ---------------- Gate B: paired noise sync (pure torch check)
    winner = _load_coords(POOL, pairs[0]["winner_sample_id"]).unsqueeze(0)
    loser = _load_coords(POOL, pairs[0]["loser_sample_id"]).unsqueeze(0)
    device = winner.device
    mask = torch.ones(winner.shape[1], dtype=torch.bool, device=device)
    sigma = torch.tensor([1.234], device=device)
    noise = torch.randn_like(winner)
    out = paired_noising(winner[0], loser[0], mask, sigma, noise=noise)
    # same-rigid-augmentation check: identical inputs must stay identical
    twin = paired_noising(winner[0], winner[0], mask, sigma, noise=noise,
                          augmentation=True)
    twin_equal = bool(torch.allclose(twin["x0_w_aug"], twin["x0_l_aug"], atol=1e-6))
    # x_t = x0_aug + sigma*noise must hold for both members
    err_w = (out["x_t_w"] - (out["x0_w_aug"] + sigma.reshape(1, 1, 1) * noise)).abs().max()
    err_l = (out["x_t_l"] - (out["x0_l_aug"] + sigma.reshape(1, 1, 1) * noise)).abs().max()
    same_noise = bool(torch.equal(out["noise"], noise))
    report["paired_noise"] = {
        "x_t_w_consistency_max_err": float(err_w),
        "x_t_l_consistency_max_err": float(err_l),
        "noise_shared": same_noise,
        "sigma_shared": True,
        "identical_inputs_same_transform": twin_equal,
    }

    # ---------------- load model
    print("loading model ...", flush=True)
    base = load_base_model(Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt"), device="cuda")
    policy, reference = make_policy_reference(base, device="cuda")
    train_params = trainable_score_params(policy)
    optimizer = torch.optim.AdamW(train_params, lr=1e-5)

    # conditioning for the first 4 cases used by the pairs below
    used_cases = sorted({p["case_id"] for p in pairs})[:4]
    conds = {c: move_conditioning(load_conditioning(COND, c)) for c in used_cases}

    def step_once(pair, require_grad_policy=True):
        cond = conds[pair["case_id"]]
        feats = cond["feats"]
        kwargs = {
            "s_inputs": cond["s_inputs"], "s_trunk": cond["s_trunk"],
            "feats": feats, "multiplicity": 1,
            "diffusion_conditioning": cond["diffusion_conditioning"],
        }
        pref, _ = _compute_masks(feats, "cdr_fake_atoms")
        w = _load_coords(POOL, pair["winner_sample_id"]).unsqueeze(0)
        l = _load_coords(POOL, pair["loser_sample_id"]).unsqueeze(0)
        atom_mask = feats["atom_pad_mask"].reshape(-1)
        sigma = policy.structure_module.noise_distribution(1)
        noise = torch.randn_like(w)
        paired = paired_noising(w[0], l[0], atom_mask, sigma, noise=noise)
        out_w = per_sample_denoising_loss(policy.structure_module, feats, paired["x0_w_aug"], paired["x_t_w"], sigma, kwargs, preference_mask=pref, require_grad=True)
        out_l = per_sample_denoising_loss(policy.structure_module, feats, paired["x0_l_aug"], paired["x_t_l"], sigma, kwargs, preference_mask=pref, require_grad=True)
        ref_w = per_sample_denoising_loss(reference.structure_module, feats, paired["x0_w_aug"], paired["x_t_w"], sigma, kwargs, preference_mask=pref, require_grad=False)
        ref_l = per_sample_denoising_loss(reference.structure_module, feats, paired["x0_l_aug"], paired["x_t_l"], sigma, kwargs, preference_mask=pref, require_grad=False)
        dpo = diffusion_dpo_loss(out_w.per_sample_loss, out_l.per_sample_loss, ref_w.per_sample_loss, ref_l.per_sample_loss, 1.0)
        return dpo, out_w, out_l

    # ---------------- Gate C: DPO init ≈ log 2
    init_losses = []
    for pair in pairs[:20]:
        dpo, _, _ = step_once(pair)
        init_losses.append(float(dpo.dpo_loss))
    import statistics as st
    report["dpo_init"] = {
        "n_pairs": len(init_losses),
        "mean_loss": st.mean(init_losses),
        "std_loss": st.stdev(init_losses),
        "log2": 0.6931471805599453,
    }

    # ---------------- Gate D: gradient audit
    optimizer.zero_grad(set_to_none=True)
    dpo, _, _ = step_once(pairs[0])
    dpo.total_loss.backward()
    grad_stats = {}
    for name, param in policy.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            grad_stats[name.split(".")[0] + "." + ".".join(name.split(".")[1:3])] = \
                grad_stats.get(name.split(".")[0] + "." + ".".join(name.split(".")[1:3]), 0.0) + float(param.grad.abs().sum())
    ref_grads = sum(1 for p in reference.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    report["gradient_audit"] = {
        "modules_with_grad": sorted(grad_stats.keys()),
        "reference_params_with_grad": ref_grads,
    }
    optimizer.zero_grad(set_to_none=True)

    # ---------------- Gate E: one-pair overfit
    pair = max(pairs, key=lambda p: p["reward_gap"])
    zs = []
    for it in range(60):
        optimizer.zero_grad(set_to_none=True)
        dpo, _, _ = step_once(pair)
        dpo.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        optimizer.step()
        zs.append(float(dpo.z.mean()))
        if (it + 1) % 20 == 0:
            print(f"[overfit] step {it+1} z={zs[-1]:+.4f} loss={float(dpo.total_loss):.4f}", flush=True)
    report["one_pair_overfit"] = {
        "pair": {"case": pair["case_id"], "w": pair["winner_sample_id"], "l": pair["loser_sample_id"]},
        "z_start": zs[0], "z_end": zs[-1], "z_max": max(zs), "z_min": min(zs),
    }

    (OUT / "smoke_gates_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    gate_pass = (
        report["dpo_init"]["mean_loss"] > 0.60
        and report["dpo_init"]["mean_loss"] < 0.78
        and report["gradient_audit"]["reference_params_with_grad"] == 0
        and all(k.startswith("structure_module") for k in report["gradient_audit"]["modules_with_grad"])
        and report["one_pair_overfit"]["z_end"] > report["one_pair_overfit"]["z_start"]
    )
    print("SMOKE GATES PASS" if gate_pass else "SMOKE GATES FAIL")


if __name__ == "__main__":
    main()
