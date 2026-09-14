#!/usr/bin/env python
"""Phase A1/A2: batch-score all counterfactual sequences with the CDR classifier.

Uses the existing scorer worker (task book §11/§57) with a dedicated cache at
runs/next_stage/scorer_cache.sqlite.  Scores are raw `cdr_camelid_margin`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cli.common import case_spec_for  # noqa: E402
from vhh_rl.data.case import RLCase  # noqa: E402
from vhh_rl.rewards.scoring import ScorerAdapter  # noqa: E402

POOL = ROOT / "runs/native_pool"
OUT = ROOT / "runs/next_stage/counterfactual"
CACHE = ROOT / "runs/next_stage/scorer_cache.sqlite"

WORKER_PY = "/share/home/rongdingyi/.conda/envs/vhh-guidance-esmc/bin/python"
WORKER_SCRIPT = str(ROOT / "src/vhh_rl/rewards/scorer_worker.py")
SCORER_ROOT = "/share/home/rongdingyi/programs/proteingen/vhh_guidance"


def load_cases() -> dict[str, RLCase]:
    cases = {}
    for line in (ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl").open():
        row = json.loads(line)
        if row.get("split") != "train":
            continue
        cases[row["case_id"]] = RLCase(
            case_id=row["case_id"], structure_path=Path(row["structure_path"]),
            chain_id=row.get("chain_id", "A"), full_sequence=row["full_sequence"],
            design_positions=tuple(row["design_positions"]),
            fr_positions=tuple(row["fr_positions"]), split="train",
            seed_base=int(row.get("seed_base", 0)),
        )
    return cases


def main() -> None:
    cases = load_cases()
    rows = [json.loads(l) for l in (OUT / "counterfactual_sequences.jsonl").open()]
    by_case: dict[str, list[dict]] = {}
    for r in rows:
        by_case.setdefault(r["case_id"], []).append(r)

    adapter = ScorerAdapter(
        worker_python=WORKER_PY, worker_script=WORKER_SCRIPT,
        scorer_root=SCORER_ROOT, objective="cdr_margin_guarded",
        cache_path=CACHE,
    )
    scores: dict[str, float] = {}
    for cid, rs in sorted(by_case.items()):
        unique = {}
        for r in rs:
            unique.setdefault(r["sequence_sha1"], r["sequence"])
        shas = list(unique.keys())
        seqs = [unique[s] for s in shas]
        batch = adapter.score_sequences(seqs, spec=case_spec_for(cases[cid]))
        for s, v in zip(shas, batch.raw_scores.tolist()):
            scores[s] = float(v)
        print(f"[cf-score] {cid}: {len(seqs)} sequences", flush=True)
    adapter.close()

    with (OUT / "counterfactual_scores.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps({
                "cf_id": r["cf_id"], "pair_id": r["pair_id"], "case_id": r["case_id"],
                "kind": r["kind"], "position": r["position"], "region": r["region"],
                "sequence_sha1": r["sequence_sha1"], "score": scores[r["sequence_sha1"]],
            }) + "\n")
    manifest = {
        "n_scored_sequences": len(scores),
        "objective": "cdr_margin_guarded (scores = cdr_camelid_margin)",
        "scorer_root": SCORER_ROOT,
        "worker_python": WORKER_PY,
        "cache": str(CACHE),
    }
    (OUT / "score_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
