#!/usr/bin/env python
"""Classifier panel for M3-v2 selected candidates (combines with unified scores).

Scores the M3-v2 pool's 8 selected sequences per case with the frozen
VHH-source ensemble (same fields as the native panel).
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
GUID = Path("/share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance")
SCORER_ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_guidance")
OUT_DIR = ROOT / "runs/report_unified"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCORER_ROOT / "src"))
from vhh_guidance.adapters.vhh_source_adapter import VHHSourceAdapter  # noqa: E402
from vhh_guidance.schemas import VHHSpec  # noqa: E402
from vhh_rl.rewards.scorer_worker import build_spec  # noqa: E402
from vhh_rl.cli.common import cdr_range  # noqa: E402
from vhh_rl.data.case import RLCase  # noqa: E402


def spec_for(row: dict) -> dict:
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


def main() -> None:
    manifest = {}
    for line in (ROOT / "runs/round1_rl_split/rl_manifest_valid100test.jsonl").open():
        row = json.loads(line)
        manifest[row["case_id"]] = row

    config = yaml.safe_load((SCORER_ROOT / "configs" / "models.local.yaml").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = VHHSourceAdapter(config, device)
    scorer.load()

    out = open(OUT_DIR / "m3v2_classifier.jsonl", "w")
    n = 0
    with torch.no_grad():
        for case_dir in sorted((GUID / "outputs/main_boltzgen/m3v2_pool/pool").iterdir()):
            if not case_dir.is_dir():
                continue
            payload = json.load(open(case_dir / "g0_selection.json"))
            cid = payload["case_id"]
            seqs = [c["sequence"] for c in payload["candidates"] if c.get("selected")]
            if not seqs:
                continue
            spec = build_spec(spec_for(manifest[cid]), VHHSpec)
            results = scorer.score_sequences(seqs, spec)
            for seq, res in zip(seqs, results):
                out.write(json.dumps({"arm": "m3v2", "case_id": cid, "sequence": seq,
                                      **res.as_dict()}) + "\n")
            n += len(seqs)
    out.close()

    rows = [json.loads(l) for l in open(OUT_DIR / "m3v2_classifier.jsonl")]
    fields = ["camelid_native_likeness_score", "cdr_camelid_margin", "p_cdr_camelid",
              "background_discordance_jsd", "nativeness_ensemble_js", "cdr_energy",
              "cdr_embedding_distance"]
    summary = {f: st.mean(r[f] for r in rows) for f in fields}
    summary["n"] = len(rows)
    summary["ood_review_rate"] = sum(1 for r in rows if r["ood_flag"] == "review") / len(rows)
    (OUT_DIR / "m3v2_classifier_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
