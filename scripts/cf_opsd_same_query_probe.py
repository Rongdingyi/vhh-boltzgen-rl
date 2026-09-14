#!/usr/bin/env python
"""Phase C: same-query finite realization probe (task book §35-§45)."""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

import torch

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cf_opsd.checkpoint import (  # noqa: E402
    load_base_model, make_policy_reference, save_native_checkpoint, trainable_score_params,
)
from vhh_rl.cf_opsd.same_query_fit import distance, student_prediction  # noqa: E402
from vhh_rl.cli.common import case_spec_for  # noqa: E402
from vhh_rl.native_atom14.decode import decode_atom14, fr_check, sequence_from_feat  # noqa: E402
from vhh_rl.native_atom14.dpo_trainer import load_conditioning, move_conditioning  # noqa: E402
from vhh_rl.native_atom14.reward import make_reward_adapter  # noqa: E402

BASE = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt")
TARGETS = ROOT / "runs/cf_opsd/target_probe/targets.pt"
OUT = ROOT / "runs/cf_opsd/realization_probe"
DOC = ROOT / "docs/cf_opsd/CF_OPSD_REALIZATION_PROBE.md"
STEPS = (1, 5, 20)
VARIANTS = {"V0-A": "target_mask_only", "V0-B": "target_mask_weak_hold"}
DEVICE = "cuda"


def decode_seq(coords, feats):
    feat = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in feats.items()}
    feat["coords"] = coords.clone()
    out = decode_atom14(feat)
    seq, _tok, invalid = sequence_from_feat(out)
    return seq, invalid


def main() -> None:

    gate_b_path = ROOT / "runs/cf_opsd/target_probe/gate_b.json"
    if gate_b_path.is_file():
        _gb = json.loads(gate_b_path.read_text())
        if _gb.get("selected_radius") is None:
            OUT.mkdir(parents=True, exist_ok=True)
            DOC.write_text("# CF-OPSD Realization Probe（Phase C）\n\n"
                           "Gate B FAIL（无通过半径）→ 按任务书 §32/§95 停止，Phase C 不执行。\n")
            print("Gate B FAIL -> Phase C not run (task book §32)")
            return
    targets = torch.load(TARGETS, map_location="cpu", weights_only=False)
    targets = [t for t in targets if t["control"] == "cf"]
    if not targets:
        raise SystemExit("no CF targets from target probe")
    base = load_base_model(BASE, device=DEVICE)
    student, _ref = make_policy_reference(base, device=DEVICE)
    pristine = {k: v.detach().clone() for k, v in student.state_dict().items()}
    params = trainable_score_params(student)
    scorer = make_reward_adapter(cache_path=ROOT / "runs/cf_opsd/reward_cache.sqlite")
    cond_dir = ROOT / "runs/native_pool/conditioning"

    rows = []
    for ti, rec in enumerate(targets):
        cid = rec["case_id"]
        cond = move_conditioning(load_conditioning(cond_dir, cid))
        kwargs = {"s_inputs": cond["s_inputs"], "s_trunk": cond["s_trunk"],
                  "feats": cond["feats"], "multiplicity": 1,
                  "diffusion_conditioning": cond["diffusion_conditioning"]}
        feats = cond["feats"]
        query = rec["query_coords"].to(DEVICE).float().unsqueeze(0)
        sigma = torch.tensor([float(rec["sigma"])], device=DEVICE)
        target = rec["target_coords"].to(DEVICE).float().unsqueeze(0)
        anchor = rec["anchor_coords"].to(DEVICE).float().unsqueeze(0)
        anchor_seq = rec["anchor_sequence"]
        anchor_reward = rec["anchor_reward"]
        target_reward = rec["target_reward"]
        target_gain = (target_reward - anchor_reward) if (target_reward is not None
                                                          and anchor_reward is not None) else None
        # masks (same construction as static trainer)
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
        resolved = feats["atom_resolved_mask"].reshape(-1).bool()
        hold_mask = (pad & resolved & ~mask).to(DEVICE)

        for vname, variant in VARIANTS.items():
            student.load_state_dict(pristine)
            optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=0.0)
            checkpoints = {0: student_prediction(student.structure_module, query, sigma, kwargs)}
            for step in range(1, max(STEPS) + 1):
                optimizer.zero_grad(set_to_none=True)
                denoised, _ = student.structure_module.preconditioned_network_forward(
                    query, sigma, training=False, network_condition_kwargs=kwargs)
                diff = (denoised.float() - target) ** 2
                loss = diff[:, mask, :].sum(dim=-1).mean()
                if variant == "target_mask_weak_hold":
                    hold = ((denoised.float() - anchor) ** 2)[:, hold_mask, :].sum(dim=-1).mean()
                    loss = loss + 0.05 * hold
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                if step in STEPS:
                    checkpoints[step] = student_prediction(
                        student.structure_module, query, sigma, kwargs)

            for step in (0,) + STEPS:
                pred = checkpoints[step]
                seq, invalid = decode_seq(pred[0].cpu(), rec["feats_common"])
                reward = None
                if not invalid:
                    batch = scorer.score_sequences([seq], spec=case_spec_for_by_id(cid, scorer))
                    reward = float(batch.raw_scores[0])
                realized = (reward - anchor_reward) if (reward is not None
                                                        and anchor_reward is not None) else None
                ratio = (realized / target_gain) if (realized is not None and target_gain
                                                     and abs(target_gain) > 1e-8) else None
                rows.append({
                    "case_id": cid, "variant": vname, "step": step,
                    "anchor_reward": anchor_reward, "target_reward": target_reward,
                    "prediction_reward": reward, "target_gain": target_gain,
                    "realized_gain": realized, "realization_ratio": ratio,
                    "dist_pred_target": distance(pred[0], target[0], mask),
                    "dist_pred_anchor": distance(pred[0], anchor[0], mask),
                    "invalid": invalid,
                })
                print(f"[{cid}][{vname}] step {step}: pred_reward={reward} ratio={ratio}",
                      flush=True)

    scorer.close()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "same_query_metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def agg(variant: str, step: int) -> dict:
        rr = [r for r in rows if r["variant"] == variant and r["step"] == step]
        gains = [r["realized_gain"] for r in rr if r["realized_gain"] is not None]
        ratios = [r["realization_ratio"] for r in rr if r["realization_ratio"] is not None]
        invalid = sum(1 for r in rr if r["invalid"])
        return {"n": len(rr), "positive": sum(1 for g in gains if g > 0),
                "median_gain": st.median(gains) if gains else None,
                "median_ratio": st.median(ratios) if ratios else None,
                "invalid": invalid}

    summary = {v: {s: agg(v, s) for s in (0,) + STEPS} for v in VARIANTS}
    gate_c = {}
    for v in VARIANTS:
        a = summary[v][20]
        gate_c[v] = (a["median_gain"] is not None and a["median_gain"] > 0
                     and a["positive"] >= 3 and (a["median_ratio"] or 0) >= 0.30
                     and a["invalid"] <= 0.05 * a["n"])
    (OUT / "gate_c.json").write_text(json.dumps({"summary": summary, "gate_c": gate_c}, indent=1))

    table = "\n".join(
        f"| {v} | {s} | {summary[v][s]['positive']}/{summary[v][s]['n']} | "
        f"{fmt(summary[v][s]['median_gain'])} | {fmt(summary[v][s]['median_ratio'])} | "
        f"{summary[v][s]['invalid']} |"
        for v in VARIANTS for s in (0,) + STEPS)
    doc = f"""# CF-OPSD Realization Probe（Phase C）

## Same-query realization（4 targets, q* 见 TARGET_PROBE）

| variant | step | positive | median realized gain | median ratio | invalid |
|---|---|---|---|---|---|
{table}

- Gate C（20 step）：V0-A={'PASS' if gate_c['V0-A'] else 'FAIL'}，
  V0-B={'PASS' if gate_c['V0-B'] else 'FAIL'}
- 如果两者相近：选择 **V0-A**（更简单）；如果 V0-A 漂移而 V0-B 稳定，用 V0-B。
"""
    DOC.write_text(doc)
    print(json.dumps(gate_c, indent=1))
    print(f"wrote {DOC}")


def fmt(v):
    return "n/a" if v is None else f"{v:+.3f}"


def case_spec_for_by_id(cid, scorer):
    from pathlib import Path as _P
    from vhh_rl.data.case import read_manifest
    cases = {c.case_id: c for c in read_manifest(
        ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl", splits=("train", "test"))}
    return case_spec_for(cases[cid])


if __name__ == "__main__":
    main()
