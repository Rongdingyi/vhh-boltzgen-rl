#!/usr/bin/env python3
"""Design-model (atom14 readout) arm: extract sequences + score full classifier panel.

The BoltzGen design step (all-atom atom14 diffusion) writes residue identities
into backbone.cif via geometric decoding (res_from_atom14). Those readout
sequences were never scored; this adds them as a new arm ("design") to the
5-arm comparison using the same frozen VHH-source ensemble.

Also writes an extraction sanity report: FR positions must match the manifest
(case natives), and the sequence must match boltzgen_run/intermediate_designs/
design.cif (the exact file the IF step consumed).
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import gemmi
import torch
import yaml

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
SCORER_ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_guidance")
MANIFEST = ROOT / "runs/round1_rl_split/rl_manifest_valid100test.jsonl"
OUT = ROOT / "runs/round1_rl_split/design_arm_scores.jsonl"
REPORT = ROOT / "runs/round1_rl_split/design_arm_extract_report.json"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCORER_ROOT / "src"))
from vhh_guidance.adapters.vhh_source_adapter import VHHSourceAdapter  # noqa: E402
from vhh_guidance.schemas import VHHSpec  # noqa: E402
from vhh_rl.rewards.scorer_worker import build_spec  # noqa: E402
from vhh_rl.cli.common import cdr_range  # noqa: E402
from vhh_rl.data.case import RLCase  # noqa: E402


def read_seq(path: Path, expected_len: int) -> tuple[str, str]:
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    model = st[0]
    best = None
    for ch in model:
        seq = gemmi.one_letter_code([r.name for r in ch]).upper()
        if len(seq) == expected_len:
            nx = seq.count("X")
            if best is None or nx < best[2]:
                best = (ch.name, seq, nx)
    if best is None:
        cand = [(ch.name, len(gemmi.one_letter_code([r.name for r in ch]))) for ch in model]
        raise RuntimeError(f"{path}: no chain of length {expected_len}; chains={cand}")
    return best[0], best[1]


def main() -> None:
    rows = [json.loads(l) for l in open(MANIFEST)]
    seqs = {}
    per_case = {}
    n_fr_diff_cases = 0
    n_cross_ok = n_cross_bad = 0
    for row in rows:
        cid = row["case_id"]
        chain, seq = read_seq(Path(row["structure_path"]), len(row["full_sequence"]))
        design = set(row["design_positions"])
        fr_diffs = [i for i in range(len(seq)) if i not in design and seq[i] != row["full_sequence"][i]]
        design_diffs = [i for i in design if seq[i] != row["full_sequence"][i]]
        n_gly = sum(1 for i in design if seq[i] == "G")
        d2 = Path(row["structure_path"]).parent / "boltzgen_run/intermediate_designs/design.cif"
        cross = None
        if d2.is_file():
            try:
                _, seq2 = read_seq(d2, len(seq))
                cross = (seq2 == seq)
            except Exception as exc:  # noqa: BLE001
                cross = f"error:{exc}"
        if cross is True:
            n_cross_ok += 1
        elif cross is False:
            n_cross_bad += 1
        if fr_diffs:
            n_fr_diff_cases += 1
        per_case[cid] = {
            "chain": chain,
            "len": len(seq),
            "n_fr_diffs": len(fr_diffs),
            "n_design_diffs": len(design_diffs),
            "n_gly_at_design": n_gly,
            "cross_check_design_cif": cross,
            "n_x": seq.count("X"),
        }
        seqs[cid] = seq
    summary = {
        "cases": len(rows),
        "cases_with_fr_diffs": n_fr_diff_cases,
        "cross_check_ok": n_cross_ok,
        "cross_check_fail": n_cross_bad,
        "cases_with_x": sum(1 for v in per_case.values() if v["n_x"] > 0),
        "mean_gly_at_design": sum(v["n_gly_at_design"] for v in per_case.values()) / len(rows),
    }
    REPORT.write_text(json.dumps({"summary": summary, "per_case": per_case}, indent=1))
    print(json.dumps(summary, indent=1), flush=True)

    config = yaml.safe_load((SCORER_ROOT / "configs" / "models.local.yaml").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = VHHSourceAdapter(config, device)
    scorer.load()

    def spec_for(row):
        case = RLCase(
            case_id=row["case_id"],
            structure_path=Path(row["structure_path"]),
            chain_id=row.get("chain_id", "A"),
            full_sequence=row["full_sequence"],
            design_positions=tuple(row["design_positions"]),
            fr_positions=tuple(row.get("fr_positions", ())),
            split=row.get("split", "test"),
            seed_base=int(row.get("seed_base", 0)),
        )
        return {
            "sequence": case.full_sequence,
            "design_positions": list(case.design_positions),
            "cdr1": list(cdr_range(case, "cdr1")),
            "cdr2": list(cdr_range(case, "cdr2")),
            "cdr3": list(cdr_range(case, "cdr3")),
            "chain_id": case.chain_id,
        }

    out = open(OUT, "w")
    done = 0
    with torch.no_grad():
        for row in rows:
            cid = row["case_id"]
            spec = build_spec(spec_for(row), VHHSpec)
            res = scorer.score_sequences([seqs[cid]], spec)[0]
            out.write(json.dumps({"case_id": cid, "arm": "design", "idx": 0,
                                  "sequence": seqs[cid], **res.as_dict()}) + "\n")
            done += 1
            if done % 25 == 0:
                print(f"  scored {done}/{len(rows)}", flush=True)
    out.close()
    print(f"DONE {done} rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
