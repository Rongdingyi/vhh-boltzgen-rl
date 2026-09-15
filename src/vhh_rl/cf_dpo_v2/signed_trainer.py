"""CF-DPO v2 signed / full-v2 trainer (proposal §8).

Edge loss (single training objective)::

    z_e = [H_theta(X_b) - H_theta(X_a)] / tau
    t_e = sigmoid(dR_e / tau)
    L   = BCEWithLogits(z_e, t_e)

with the reference-relative potential approximated by the audited denoising
energy proxy ``H ≈ -kappa (ell_theta - ell_0)``.  ``ell`` uses exactly the
native noising distribution / preconditioning / coordinate error weights from
``native_atom14.denoise_loss``.  No scorer gradients, no EMA behavior, no
on-policy rollout (proposal §11.1).
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

from ..native_atom14.checkpoint import (
    load_base_model, make_policy_reference, parameter_drift,
    save_native_checkpoint, trainable_score_params,
)
from ..native_atom14.denoise_loss import per_sample_denoising_loss
from ..native_atom14.dpo_trainer import load_conditioning, move_conditioning
from ..native_atom14.paired_noise import paired_noising
from .sampling import choose_edge

DEVICE = "cuda"
ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")


def _log(path: Path, record: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def edge_loss(policy_sm, ref_sm, feats: dict, coords_a: torch.Tensor,
              coords_b: torch.Tensor, sigma: torch.Tensor, noise: torch.Tensor,
              kwargs: dict, *, tau: float, kappa: float,
              require_grad: bool) -> tuple[torch.Tensor, dict]:
    atom_mask = feats["atom_pad_mask"]
    if atom_mask.dim() > 1:
        atom_mask = atom_mask.reshape(-1)
    paired = paired_noising(
        coords_a, coords_b, atom_mask, sigma,
        augmentation=policy_sm.coordinate_augmentation, noise=noise,
    )
    la = per_sample_denoising_loss(policy_sm, feats, paired["x0_w_aug"], paired["x_t_w"],
                                   sigma, kwargs, require_grad=require_grad)
    lb = per_sample_denoising_loss(policy_sm, feats, paired["x0_l_aug"], paired["x_t_l"],
                                   sigma, kwargs, require_grad=require_grad)
    ra = per_sample_denoising_loss(ref_sm, feats, paired["x0_w_aug"], paired["x_t_w"],
                                   sigma, kwargs, require_grad=False)
    rb = per_sample_denoising_loss(ref_sm, feats, paired["x0_l_aug"], paired["x_t_l"],
                                   sigma, kwargs, require_grad=False)
    h_b = -kappa * (lb.per_sample_loss - rb.per_sample_loss)
    h_a = -kappa * (la.per_sample_loss - ra.per_sample_loss)
    z = (h_b - h_a) / tau
    info = {
        "h_a": float(h_a.detach().mean()), "h_b": float(h_b.detach().mean()),
        "ell_b": float(lb.per_sample_loss.detach().mean()),
        "ell_ref_b": float(rb.per_sample_loss.detach().mean()),
    }
    return z.reshape(1), info


def run_signed(base_checkpoint: str | Path, graph_path: str | Path,
               output_dir: str | Path, *, variant: str = "v2", tau: float = 1.0,
               kappa: float = 1.0, updates: int = 100, lr: float = 1e-5,
               max_grad_norm: float = 1.0, checkpoint_every: int = 50,
               same_seq_ratio: float = 0.25, seed: int = 20260914,
               conditioning_dir: str | Path | None = None,
               n_sigma: int = 3, calib_steps: int = 10, target_h_std: float = 1.0,
               log_tag: str = "cfd2") -> dict:
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")  # match native sampling context
    random.seed(seed)
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    nodes, edges = graph["nodes"], graph["edges"]
    if variant == "signed":
        edges = [e for e in edges if e["kind"] != "same_seq"]
    if not edges:
        raise RuntimeError("empty comparison graph")

    base = load_base_model(base_checkpoint, device=DEVICE)
    policy, reference = make_policy_reference(base, device=DEVICE)
    params = trainable_score_params(policy)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    cond_dir = Path(conditioning_dir) if conditioning_dir else None
    cond_cache: dict[str, dict] = {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    t0 = time.time()
    calib_h: list[float] = []
    kappa_eff = kappa
    for step in range(1, updates + 1):
        edge = choose_edge(edges, variant, same_seq_ratio, random)
        node_a, node_b = nodes[edge["a"]], nodes[edge["b"]]
        cid = node_a["case_id"]
        if cid not in cond_cache:
            cond = None
            rollout_dir = ROOT / "runs/cf_opsd/rollouts/train" / cid
            candidates = sorted(rollout_dir.glob("seed*.pt")) if rollout_dir.is_dir() else []
            if candidates:
                payload = torch.load(candidates[0], map_location="cpu", weights_only=False)
                cond = payload.get("cond_kwargs") if isinstance(payload, dict) else None
            if cond:
                cond_cache[cid] = move_conditioning(cond, device=DEVICE)
            elif cond_dir is not None:
                cond_cache[cid] = move_conditioning(
                    load_conditioning(cond_dir, cid), device=DEVICE)
            else:
                raise FileNotFoundError(
                    f"no conditioning available for {cid} (no rollout capture and no dir)")
        cond = cond_cache[cid]
        feats = cond["feats"]
        kwargs = {"s_inputs": cond["s_inputs"], "s_trunk": cond["s_trunk"],
                  "feats": feats, "multiplicity": 1,
                  "diffusion_conditioning": cond["diffusion_conditioning"]}
        coords_a = node_a["coords"].to(DEVICE).float()
        coords_b = node_b["coords"].to(DEVICE).float()
        optimizer.zero_grad(set_to_none=True)
        zs, infos = [], []
        for _draw in range(max(1, n_sigma)):
            sigma = policy.structure_module.noise_distribution(1)
            noise = torch.randn_like(coords_a.unsqueeze(0))
            z_draw, info_draw = edge_loss(
                policy.structure_module, reference.structure_module, feats,
                coords_a, coords_b, sigma, noise, kwargs, tau=tau,
                kappa=kappa_eff, require_grad=True)
            zs.append(z_draw)
            infos.append(info_draw)
        z = torch.stack(zs, dim=0).mean(dim=0)  # keep shape [1]
        info = {k: sum(i[k] for i in infos) / len(infos) for k in infos[0]}
        t = torch.sigmoid(torch.tensor([edge["dR"] / tau], device=DEVICE))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, t)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        if not math.isfinite(float(grad_norm)):
            raise RuntimeError(f"non-finite grad norm at step {step}")
        optimizer.step()
        if step <= calib_steps:
            calib_h.extend([info["h_a"], info["h_b"]])
            if step == calib_steps and len(calib_h) > 1:
                import statistics as _st
                emp_std = _st.stdev(calib_h) or 1e-8
                kappa_eff = kappa * (target_h_std / emp_std)
                print(f"[{log_tag}] calibrated kappa: emp_std={emp_std:.3e} "
                      f"-> kappa_eff={kappa_eff:.3e}", flush=True)
        record = {
            "step": step, "edge_kind": edge["kind"], "case_id": cid,
            "dR": edge["dR"], "t": float(t), "z": float(z.detach()),
            "loss": float(loss.detach()), "grad_norm": float(grad_norm),
            "kappa_eff": float(kappa_eff), "n_sigma": int(n_sigma),
            "seconds": time.time() - t0, **info,
        }
        if step % 10 == 0 or step == 1:
            drift = parameter_drift(policy, reference)
            record["param/drift_total"] = drift["total"]
        history.append(record)
        _log(output_dir / "train_metrics.jsonl", record)
        if step % 10 == 0 or step == 1:
            print(f"[{log_tag}] step {step}/{updates} kind={edge['kind']} "
                  f"dR={edge['dR']:+.3f} z={record['z']:+.4f} loss={record['loss']:.4f} "
                  f"grad={grad_norm:.3f}", flush=True)
        if checkpoint_every and (step % checkpoint_every == 0 or step == updates):
            ckpt = output_dir / f"checkpoint_{step:04d}.pt"
            save_native_checkpoint(base_checkpoint, policy, ckpt, {
                "method": f"cf_dpo_v2_{variant}", "step": step, "tau": tau,
                "kappa": kappa, "graph": str(graph_path), "seed": seed,
            })
            print(f"[{log_tag}] saved {ckpt.name}", flush=True)
    summary = {"variant": variant, "updates": updates, "n_edges": len(edges),
               "n_nodes": len(nodes), "kappa_eff_final": float(kappa_eff),
               "n_sigma": int(n_sigma),
               "final_z": history[-1]["z"], "final_loss": history[-1]["loss"],
               "elapsed_seconds": time.time() - t0}
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=1))
    return summary
