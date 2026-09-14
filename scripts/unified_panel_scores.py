#!/usr/bin/env python
"""Unified panel scoring: ESM-C CDR PLL + ProteinMPNN CDR NLL + recovery.

Computes the same three sequence-level metrics for every method/arm in the
combined report (G0/M1/M2/M3/M3-v2/IF-RL/native-RL), on the same valid100
registry and the same frozen canonical backbone.

Outputs: runs/report_unified/unified_scores.jsonl + unified_summary.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
GUID = Path("/share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance")
UNI = GUID / "outputs/unified_valid100__g1_g2_g6__m1_m2_m3__g0_512_v1"
OUT = ROOT / "runs/report_unified"

sys.path.insert(0, str(GUID / "src"))

MPNN_REPO = Path("/share/home/rongdingyi/programs/proteingen/ProteinMPNN")
MPNN_CKPT = MPNN_REPO / "vanilla_model_weights/v_48_020.pt"
ESMC_MODEL = Path("/share/data/limc/esmc-600m-vhh")
TF_ROOT = GUID / "outputs/target_free_strict8_v1"


def load_registry() -> dict[str, dict]:
    reg = {}
    with (GUID / "data_manifests/valid100.csv").open() as fh:
        for row in csv.DictReader(fh):
            cid = row["case_id"].strip()
            design = (
                json.loads(row["cdr1_positions"])
                + json.loads(row["cdr2_positions"])
                + json.loads(row["cdr3_positions"])
            )
            design = sorted(int(p) for p in design)
            native = row["sequence"].strip().upper()
            fixed = [i for i in range(len(native)) if i not in set(design)]
            backbone = TF_ROOT / "backbones" / cid / "replicate_00" / "backbone.cif"
            reg[cid] = {
                "native": native,
                "design_positions": design,
                "fixed_positions": fixed,
                "backbone": backbone,
                "vhh_chain": row.get("vhh_chain", "A"),
            }
    return reg


def collect_sources() -> list[tuple[str, str, str]]:
    """Return (method, case_id, sequence) rows for every arm."""
    rows: list[tuple[str, str, str]] = []

    # G0 top-8 (selected_in_top8)
    import glob
    for fp in sorted(glob.glob(str(GUID / "outputs/cdr_all/g0_t1.00/g0_scores/g0_scores_shard*.csv"))):
        for r in csv.DictReader(open(fp)):
            if str(r.get("selected_in_top8", "")).strip().lower() == "true":
                rows.append(("G0", r["case_id"], r["sequence"]))

    # M1/M2/M3 from the unified candidates panel
    for r in csv.DictReader((UNI / "candidates.csv").open()):
        m = r["method"].strip()
        if m in ("M1", "M2", "M3"):
            rows.append((m, r["case_id"], r["sequence"]))

    # M3-v2 selected candidates
    for case_dir in sorted((GUID / "outputs/main_boltzgen/m3v2_pool/pool").iterdir()):
        if not case_dir.is_dir():
            continue
        payload = json.load(open(case_dir / "g0_selection.json"))
        for c in payload["candidates"]:
            if c.get("selected"):
                rows.append(("M3v2", payload["case_id"], c["sequence"]))

    # IF-RL arms
    for line in open(ROOT / "runs/round1_rl_split/arm3_scores.jsonl"):
        r = json.loads(line)
        m = {"base": "baseIF", "rl": "IF-RL_R1", "r2s1": "IF-RL_R2"}.get(r["arm"])
        if m:
            rows.append((m, r["case_id"], r["sequence"]))

    # Native arms
    for line in open(ROOT / "runs/native_pool/native_panel_valid100.jsonl"):
        r = json.loads(line)
        m = {"base": "nativeBase", "n1": "N1", "n2": "N2", "n3": "N3", "n4": "N4"}[r["arm"]]
        rows.append((m, r["case_id"], r["sequence"]))
    return rows


def recovery(sequence: str, native: str, design: list[int]) -> float:
    return sum(sequence[i] == native[i] for i in design) / len(design)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reg = load_registry()
    rows = collect_sources()
    import collections
    print("[unified] source rows:", collections.Counter(m for m, _, _ in rows), flush=True)

    import torch
    from vhh_esmc_guidance.adapters.vhh_esmc import VHHESMCAdapter
    from vhh_esmc_guidance.sequence.pll import masked_pll_details
    from vhh_esmc_guidance.adapters.proteinmpnn import ProteinMPNNAdapter, AA_TO_MPNN

    device = "cuda"
    adapter_esm = VHHESMCAdapter.from_pretrained(ESMC_MODEL, device=device, torch_dtype=torch.bfloat16)
    adapter_esm.model.to(dtype=torch.bfloat16)
    adapter_mpnn = ProteinMPNNAdapter(str(MPNN_REPO), str(MPNN_CKPT), device=device)
    adapter_mpnn.load()

    # prepare each case's frozen backbone once
    prepared: dict[str, object] = {}
    for cid, case in reg.items():
        template = list(case["native"])
        for pos in case["design_positions"]:
            template[pos] = "A"
        prepared[cid] = adapter_mpnn.prepare_structure(
            case["backbone"], vhh_chain=case["vhh_chain"],
            vhh_sequence="".join(template),
            design_positions=case["design_positions"],
            fixed_positions=case["fixed_positions"], target_chains=(),
        )
    print("[unified] backbones prepared:", len(prepared), flush=True)

    out = open(OUT / "unified_scores.jsonl", "w")
    n = 0
    with torch.inference_mode():
        for method, cid, seq in rows:
            case = reg[cid]
            rec = recovery(seq, case["native"], case["design_positions"])
            cdr_pll = float(masked_pll_details(adapter_esm, seq, positions=case["design_positions"], batch_size=64).score)
            full_pll = float(masked_pll_details(adapter_esm, seq, positions=None, batch_size=64).score)
            log_probs = adapter_mpnn.conditional_log_probs(
                seq, case["design_positions"], temperature=1.0, prepared=prepared[cid])
            selected = [float(log_probs[i, AA_TO_MPNN[seq[p]]]) for i, p in enumerate(case["design_positions"])]
            mpnn_nll = -sum(selected) / len(selected)
            out.write(json.dumps({
                "method": method, "case_id": cid, "sequence": seq,
                "cdr_pll": cdr_pll, "full_pll": full_pll,
                "mpnn_nll": mpnn_nll, "recovery": rec,
            }) + "\n")
            n += 1
            if n % 400 == 0:
                print(f"[unified] scored {n}/{len(rows)}", flush=True)
    out.close()

    # summary
    import statistics as st
    per = collections.defaultdict(list)
    for line in open(OUT / "unified_scores.jsonl"):
        per[json.loads(line)["method"]].append(json.loads(line))
    summary = {}
    for m, rs in per.items():
        summary[m] = {
            "n": len(rs),
            "cdr_pll": st.mean(r["cdr_pll"] for r in rs),
            "full_pll": st.mean(r["full_pll"] for r in rs),
            "mpnn_nll": st.mean(r["mpnn_nll"] for r in rs),
            "recovery": st.mean(r["recovery"] for r in rs),
        }
    (OUT / "unified_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
