#!/usr/bin/env python
"""Phase B1: diffusion trajectory sequence-emergence audit (task book §20-§28).

No training.  For 8 train + 8 held-out cases x 8 trajectories with the frozen
native base model:
  1. capture the full x0_coords_traj + sigma schedule (official sampler)
  2. decode every step's x0 with the official res_from_atom14
  3. per-step sequence metrics (identity to final, stable fraction, validity)
  4. per-step geometry metrics (FR-bb aligned RMSD: FR-bb / CDR-bb / fake atoms)
  5. classifier reward convergence (batch scorer over all step sequences)
  6. sigma_seq / sigma_90 + Gate B decision + figures + TEMPORAL_CREDIT_AUDIT.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cli.common import case_spec_for  # noqa: E402
from vhh_rl.credit.trajectory_audit import (  # noqa: E402
    cdr_identity, find_sigma_threshold, identity_all, kabsch_rmsd,
    normalized_convergence, stable_fraction,
)
from vhh_rl.data.case import RLCase  # noqa: E402
from vhh_rl.native_atom14.adapter import NativeDesignAdapter  # noqa: E402
from vhh_rl.native_atom14.decode import decode_atom14, sequence_from_feat  # noqa: E402
from vhh_rl.native_atom14.reward import make_reward_adapter  # noqa: E402

OUT = ROOT / "runs/next_stage/trajectory"
DOC = ROOT / "docs/TEMPORAL_CREDIT_AUDIT.md"


def load_cases() -> tuple[list[RLCase], list[RLCase]]:
    train, heldout = [], []
    for line in (ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl").open():
        row = json.loads(line)
        case = RLCase(
            case_id=row["case_id"], structure_path=Path(row["structure_path"]),
            chain_id=row.get("chain_id", "A"), full_sequence=row["full_sequence"],
            design_positions=tuple(row["design_positions"]),
            fr_positions=tuple(row["fr_positions"]), split=row.get("split", "train"),
            seed_base=int(row.get("seed_base", 0)),
        )
        (train if case.split == "train" else heldout).append(case)
    train.sort(key=lambda c: c.case_id)
    heldout.sort(key=lambda c: c.case_id)
    return train[:8], heldout[:8]


def squeeze(value):
    if torch.is_tensor(value):
        if value.dim() >= 1 and value.shape[0] == 1:
            return value.squeeze(0)
    return value


def slice_batch(feats: dict, b: int, batch_size: int) -> dict:
    """Slice the featurizer dict to one sample, mirroring the writer.

    The hook sees the model-batch dict (leading batch dim on batched tensors,
    plus occasional singleton dims); the official writer feeds
    ``res_from_atom14`` with batch dims removed (``prediction[k][0]``), so we
    remove the sample index and all leading singleton dims.
    """
    out = {}
    for k, v in feats.items():
        if torch.is_tensor(v):
            if v.dim() > 0 and v.shape[0] == batch_size:
                v = v[b]
            while v.dim() > 1 and v.shape[0] == 1:
                v = v.squeeze(0)
            out[k] = v
        else:
            out[k] = v
    return out


def design_masks(feats: dict) -> dict:
    atom_pad = squeeze(feats["atom_pad_mask"]).bool()
    a2t = squeeze(feats["atom_to_token"])
    token_of_atom = a2t.int().argmax(-1)
    design_mask = squeeze(feats["design_mask"]).bool()
    design_atoms = atom_pad & design_mask[token_of_atom]
    fake = squeeze(feats["fake_atom_mask"]).bool()
    bb = squeeze(feats["backbone_mask"]).bool()
    return {
        "atom_pad": atom_pad,
        "design_atoms": design_atoms,
        "fr_bb": atom_pad & ~design_atoms & bb,
        "cdr_bb": design_atoms & bb,
        "cdr_fake": design_atoms & fake,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-designs", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=0, help="0 = all (8+8)")
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    train, heldout = load_cases()
    cases = [("train", c) for c in train] + [("heldout", c) for c in heldout]
    if args.max_cases:
        cases = cases[: args.max_cases]

    adapter = NativeDesignAdapter(
        device=args.device, sampling_steps=args.sampling_steps,
        diffusion_batch_size=args.num_designs,
    )
    reward = make_reward_adapter(cache_path=ROOT / "runs/next_stage/scorer_cache.sqlite")

    metric_rows, seq_rows = [], []
    per_case_meta = {}
    for split, case in cases:
        seed = case.seed_base + 900000
        t0 = time.time()
        capture, traj = adapter.generate_capture_trajectory(
            case.structure_path.parent / "design.yaml", case.case_id,
            OUT / "runs" / case.case_id, num_designs=args.num_designs, seed=seed,
        )
        print(f"[B1] {case.case_id}: {len(capture.samples)} designs, "
              f"{traj['elapsed_seconds']:.0f}s, calls={len(traj['sample_calls'])}", flush=True)
        per_case_meta[case.case_id] = {
            "split": split, "seed": seed,
            "matches": traj["matches"],
            "elapsed_seconds": traj["elapsed_seconds"],
            "n_sample_calls": len(traj["sample_calls"]),
        }
        for sample in capture.samples:
            i = sample.sample_index
            final_seq = sample.decoded_sequence
            final_coords = sample.coords.float()
            match = traj["matches"][i]
            call = traj["sample_calls"][match["sample_call"]]
            b = match["batch_index"]
            feats = slice_batch(call["feats"], b, call["batch_size"])
            masks = design_masks(feats)
            # decode parity hard check on the final x0 estimate
            feat_final = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in feats.items()}
            feat_final["coords"] = final_coords
            seq_parity, _tok, _inv = sequence_from_feat(decode_atom14(feat_final))
            if seq_parity != final_seq:
                raise RuntimeError(
                    f"{case.case_id}/{i}: hook-feats decode parity broken: "
                    f"{seq_parity} != {final_seq}")
            seqs_this, rows_this = [], []
            for t in range(len(call["x0_coords_traj"])):
                coords_t = call["x0_coords_traj"][t][b].float()
                feat_copy = {k: (v.clone() if torch.is_tensor(v) else v)
                             for k, v in feats.items()}
                feat_copy["coords"] = coords_t
                out = decode_atom14(feat_copy)
                seq_t, _tokens, invalid = sequence_from_feat(out)
                seqs_this.append(seq_t)
                row = {
                    "case_id": case.case_id, "split": split, "design_index": i,
                    "sample_id": sample.sample_index, "step": t,
                    "sigma": call["t_hats"][t],
                    "valid": not invalid,
                    "cdr_identity": cdr_identity(seq_t, final_seq, case.design_positions),
                    "seq_identity": identity_all(seq_t, final_seq),
                    "rmsd_fr_bb": kabsch_rmsd(coords_t, final_coords, masks["fr_bb"], masks["fr_bb"]),
                    "rmsd_cdr_bb": kabsch_rmsd(coords_t, final_coords, masks["fr_bb"], masks["cdr_bb"]),
                    "rmsd_cdr_fake": kabsch_rmsd(coords_t, final_coords, masks["fr_bb"], masks["cdr_fake"]),
                }
                rows_this.append(row)
                metric_rows.append(row)
                seq_rows.append({
                    "case_id": case.case_id, "split": split, "design_index": i,
                    "step": t, "sequence": seq_t, "valid": not invalid,
                })
            for t, row in enumerate(rows_this):
                row["stable_fraction"] = stable_fraction(
                    seqs_this[t:], final_seq, case.design_positions)
        del traj, capture
        print(f"[B1] {case.case_id}: processed in {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ reward
    by_case: dict[str, list[str]] = {}
    for row in seq_rows:
        by_case.setdefault(row["case_id"], None)
    case_map = {c.case_id: c for _, c in cases}
    AA20 = set("ACDEFGHIKLMNPQRSTVWY")
    reward_by_seq: dict[str, float] = {}
    for cid in sorted(by_case):
        seqs = sorted({r["sequence"] for r in seq_rows
                       if r["case_id"] == cid and set(r["sequence"]) <= AA20})
        batch = reward.score_sequences(seqs, spec=case_spec_for(case_map[cid]))
        for s, v in zip(seqs, batch.raw_scores.tolist()):
            reward_by_seq[s] = float(v)
        print(f"[B1] reward {cid}: {len(seqs)} unique step sequences", flush=True)
    reward.close()
    for row in seq_rows:
        row["reward"] = reward_by_seq.get(row["sequence"])
    with (OUT / "trajectory_sequences.jsonl").open("w") as fh:
        for row in seq_rows:
            fh.write(json.dumps(row) + "\n")
    with (OUT / "trajectory_metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(metric_rows[0].keys()))
        w.writeheader()
        w.writerows(metric_rows)

    # ------------------------------------------------------------------ aggregate
    steps = sorted({r["step"] for r in metric_rows})
    def agg(key, fn=st.median):
        return [fn([r[key] for r in metric_rows if r["step"] == t]) for t in steps]
    sigmas = agg("sigma")
    med_cdr_id = agg("cdr_identity")
    med_seq_id = agg("seq_identity")
    valid_rate = [sum(1 for r in metric_rows if r["step"] == t and r["valid"])
                  / max(1, sum(1 for r in metric_rows if r["step"] == t)) for t in steps]
    med_stable = agg("stable_fraction")
    med_fr = agg("rmsd_fr_bb")
    med_cdrbb = agg("rmsd_cdr_bb")
    med_fake = agg("rmsd_cdr_fake")

    # reward convergence: per step, corr(R_t, R_final) and MAE across trajectories
    final_seq_by_traj = {(r["case_id"], r["design_index"]): r["sequence"]
                         for r in seq_rows if r["step"] == steps[-1]}
    final_reward = {k: reward_by_seq.get(s) for k, s in final_seq_by_traj.items()}
    seq_lookup = {(r["case_id"], r["design_index"], r["step"]): r["sequence"]
                  for r in seq_rows}
    def pearson(xs, ys):
        if len(xs) < 3:
            return None
        mx, my = st.mean(xs), st.mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        return cov / (vx * vy) ** 0.5 if vx and vy else None
    reward_corr, reward_mae = [], []
    for t in steps:
        pairs = []
        for r in metric_rows:
            if r["step"] != t:
                continue
            rt = reward_by_seq.get(seq_lookup[(r["case_id"], r["design_index"], t)])
            rf = final_reward.get((r["case_id"], r["design_index"]))
            if rt is not None and rf is not None:
                pairs.append((rt, rf))
        reward_corr.append(pearson([a for a, _ in pairs], [b for _, b in pairs]))
        reward_mae.append(st.mean(abs(a - b) for a, b in pairs) if pairs else None)

    # geometry normalized convergence (median across trajectories)
    by_traj: dict[tuple, dict[int, dict]] = {}
    for r in metric_rows:
        by_traj.setdefault((r["case_id"], r["design_index"]), {})[r["step"]] = r
    conv = {"fr_bb": [], "cdr_bb": [], "cdr_fake": []}
    for key, rmsd_key in (("fr_bb", "rmsd_fr_bb"), ("cdr_bb", "rmsd_cdr_bb"),
                          ("cdr_fake", "rmsd_cdr_fake")):
        for t in steps:
            traj_convs = []
            for series in by_traj.values():
                vals = [series[s][rmsd_key] for s in series]
                finite = [v for v in vals if v == v]
                scale = max(finite) if finite and max(finite) > 0 else 1.0
                traj_convs.append(max(0.0, 1.0 - series[t][rmsd_key] / scale))
            conv[key].append(st.median(traj_convs) if traj_convs else float("nan"))

    sigma_seq = find_sigma_threshold(sigmas, med_cdr_id, valid_rate, 0.80, 0.90)
    sigma_90 = find_sigma_threshold(sigmas, med_cdr_id, valid_rate, 0.90, 0.90)
    gate_window = [
        {"step": t, "sigma": sigmas[t], "conv_cdr_bb": conv["cdr_bb"][t],
         "cdr_identity": med_cdr_id[t]}
        for t in steps
        if conv["cdr_bb"][t] > 0.8 and med_cdr_id[t] < 0.5
    ]
    h3 = "STRONG SUPPORT" if gate_window else "WEAK"

    summary = {
        "n_cases": len(cases), "n_trajectories": len(cases) * args.num_designs,
        "sampling_steps": args.sampling_steps,
        "curves": {
            "step": steps, "sigma": sigmas,
            "median_cdr_identity": med_cdr_id, "median_seq_identity": med_seq_id,
            "valid_rate": valid_rate, "median_stable_fraction": med_stable,
            "median_rmsd_fr_bb": med_fr, "median_rmsd_cdr_bb": med_cdrbb,
            "median_rmsd_cdr_fake": med_fake,
            "norm_conv_fr_bb": conv["fr_bb"], "norm_conv_cdr_bb": conv["cdr_bb"],
            "norm_conv_cdr_fake": conv["cdr_fake"],
            "reward_corr_to_final": reward_corr, "reward_mae_to_final": reward_mae,
        },
        "sigma_seq": sigma_seq, "sigma_90": sigma_90,
        "gate_b_window_steps": gate_window,
        "h3": h3,
        "per_case": per_case_meta,
    }
    (OUT / "trajectory_summary.json").write_text(json.dumps(summary, indent=1))

    # ------------------------------------------------------------------ figures
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    log_sigma = [(float("nan") if s <= 0 else math.log(s)) for s in sigmas]
    x = list(range(len(steps)))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    panels = [
        (axes[0, 0], med_cdr_id, "median CDR identity to final", 0.80),
        (axes[0, 1], med_stable, "median stable residue fraction", None),
        (axes[0, 2], reward_corr, "corr(R_t, R_final)", None),
        (axes[1, 0], conv["fr_bb"], "normalized FR-bb convergence", 0.8),
        (axes[1, 1], conv["cdr_bb"], "normalized CDR-bb convergence", 0.8),
        (axes[1, 2], conv["cdr_fake"], "normalized fake-atom convergence", None),
    ]
    for ax, vals, title, line in panels:
        vals = [v if v is not None else float("nan") for v in vals]
        ax.plot(x, vals)
        if line is not None:
            ax.axhline(line, ls="--", color="gray", lw=1)
        if sigma_seq is not None:
            ax.axvline(sigmas.index(sigma_seq), ls=":", color="C3", lw=1)
        ax.set_title(title)
        ax.set_xlabel("diffusion step")
    fig.tight_layout()
    fig.savefig(OUT / "figures/figB1_emergence.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(log_sigma, med_cdr_id, marker=".", label="median CDR identity")
    ax.plot(log_sigma, conv["cdr_bb"], marker=".", label="norm. CDR-bb convergence")
    ax.plot(log_sigma, valid_rate, ls="--", label="valid decode rate")
    if sigma_seq is not None:
        ax.axvline(math.log(sigma_seq), ls=":", color="C3",
                   label=f"sigma_seq={sigma_seq:.2f}")
    ax.set_xlabel("log sigma (t_hat)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figures/figB1_vs_sigma.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------ doc
    md = f"""# Temporal Credit Audit（Phase B1，Gate B）

任务书：`NATIVE_ATOM14_NEXT_STAGE_CREDIT_RESEARCH_TASK.md` §20–§29
设置：8 train + 8 held-out cases × {args.num_designs} trajectories（frozen native base，
sampling_steps={args.sampling_steps}，官方 sampler/decoder，未训练）。

## 1. 关键阈值

| 量 | 值 |
|---|---|
| sigma_seq（median CDR identity >= 0.80 且 valid >= 0.90） | {sigma_seq} |
| sigma_90（identity >= 0.90） | {sigma_90} |
| Gate B window（CDR-bb conv > 0.8 且 identity < 0.5 的 step 数） | {len(gate_window)} |

**Gate B decision: {h3}**

## 2. 曲线（median across trajectories）

| step | sigma | CDR identity | stable frac | valid | FR-bb RMSD | CDR-bb RMSD | fake RMSD | reward corr |
|---|---|---|---|---|---|---|---|---|
"""
    for i in steps:
        rc = reward_corr[i]
        rc_s = f"{rc:+.3f}" if rc is not None else "n/a"
        md += (f"| {i} | {sigmas[i]:.2f} | {med_cdr_id[i]:.3f} | {med_stable[i]:.3f} | "
               f"{valid_rate[i]:.3f} | {med_fr[i]:.2f} | {med_cdrbb[i]:.2f} | "
               f"{med_fake[i]:.2f} | {rc_s} |\n")
    md += f"""
图：`runs/next_stage/trajectory/figures/figB1_emergence.png`、`figB1_vs_sigma.png`。

## 3. 结论

- backbone geometry 与 sequence identity 的相对先后见上表：gate window 步数
  {len(gate_window)} → H3 = {h3}。
- 若 STRONG：使用 §29 的 g(sigma)（tau=0.5，禁 sweep）做 temporal weighting；
  若 WEAK：temporal DPO 仅作小 ablation。

*脚本：`scripts/next_audit_trajectory.py`；指标：`src/vhh_rl/credit/trajectory_audit.py`*
"""
    DOC.write_text(md)
    print(json.dumps({"sigma_seq": sigma_seq, "sigma_90": sigma_90, "h3": h3,
                      "gate_window": len(gate_window)}, indent=1))
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
