#!/usr/bin/env python3
"""Full classifier-panel rescoring for native RL arms on valid100.

Scores the saved decoded sequences of base + N1..N4 selected-checkpoint arms
with the frozen VHH-source ensemble and dumps the complete VHHSourceResult
dict per sequence.  Used by the combined IF-RL + diffusion-RL report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
SCORER_ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_guidance")
POOL = ROOT / "runs/native_pool"
OUT = POOL / "native_panel_valid100.jsonl"
SUMMARY = POOL / "native_panel_valid100_summary.json"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCORER_ROOT / "src"))
from vhh_guidance.adapters.vhh_source_adapter import VHHSourceAdapter  # noqa: E402
from vhh_guidance.schemas import VHHSpec  # noqa: E402
from vhh_rl.rewards.scorer_worker import build_spec  # noqa: E402
from vhh_rl.cli.common import cdr_range  # noqa: E402
from vhh_rl.data.case import RLCase  # noqa: E402

ARMS = {
    "base": "valid100",
    "n1": "valid100_n1_s0250",
    "n2": "valid100_n2_s0500",
    "n3": "valid100_n3_s0350",
    "n4": "valid100_n4_s0500",
}


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

    out = open(OUT, "w")
    done = 0
    with torch.no_grad():
        for arm, dirname in ARMS.items():
            for case_dir in sorted((POOL / dirname).iterdir()):
                if not case_dir.is_dir():
                    continue
                cid = case_dir.name
                rows = [json.loads(l) for l in (case_dir / "metadata.jsonl").open()]
                seqs = [r["decoded_sequence"] for r in rows
                        if r["reward_raw"] is not None and not r["contains_UNK"]]
                if not seqs:
                    continue
                spec = build_spec(spec_for(manifest[cid]), VHHSpec)
                results = scorer.score_sequences(seqs, spec)
                for seq, res in zip(seqs, results):
                    out.write(json.dumps({
                        "arm": arm, "case_id": cid, "sequence": seq,
                        **res.as_dict(),
                    }) + "\n")
                done += len(seqs)
            print(f"[panel] {arm}: done ({done} total)", flush=True)
    out.close()

    # aggregate
    import collections
    import statistics as st

    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(OUT):
        r = json.loads(line)
        per[r["arm"]][r["case_id"]].append(r)
    fields = ["camelid_native_likeness_score", "cdr_camelid_margin", "p_cdr_camelid",
              "background_discordance_jsd", "nativeness_ensemble_js", "cdr_energy",
              "cdr_embedding_distance"]
    summary = {}
    for arm in ARMS:
        case_means = {}
        for cid, rows in per[arm].items():
            case_means[cid] = {f: st.mean(x[f] for x in rows) for f in fields}
        review = sum(1 for cid in per[arm] for x in per[arm][cid] if x["ood_flag"] == "review")
        n = sum(len(per[arm][cid]) for cid in per[arm])
        summary[arm] = {
            "n_sequences": n,
            **{f: st.mean(cm[f] for cm in case_means.values()) for f in fields},
            "ood_review_rate": review / max(n, 1),
        }
    SUMMARY.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
