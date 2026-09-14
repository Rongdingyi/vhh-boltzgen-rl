#!/usr/bin/env python
"""Phase D: matched CF-DPO-mini baseline on the same 4 train cases (§51).

Filters the round-1 pair set + CF weights to the fixed 4 train cases, then
trains 100 updates (checkpoints at 50 and 100) with the validated weighted
DPO trainer and identical optimizer settings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

BASE = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt")
CFG = ROOT / "configs/cf_opsd/fixed_cases.yaml"
OUT = ROOT / "runs/cf_opsd/cf_dpo_mini"


def main() -> None:
    cfg = yaml.safe_load(CFG.read_text())
    keep = set(cfg["train_cases"])
    pairs_all = [json.loads(l) for l in
                 (ROOT / "runs/native_pool/pairs_train.jsonl").open()]
    pairs = [p for p in pairs_all if p["case_id"] in keep]
    weights_all = json.loads((ROOT / "runs/next_stage/weights/residue_weights.json").read_text())
    weights = {"eta": weights_all["eta"], "seed": weights_all.get("seed"),
               "pairs": {k: v for k, v in weights_all["pairs"].items()
                         if v["case_id"] in keep}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pairs_train_mini.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
    (OUT / "residue_weights_mini.json").write_text(json.dumps(weights, indent=1))
    print(json.dumps({"n_pairs": len(pairs), "n_cases": len(keep),
                      "n_weight_pairs": len(weights["pairs"])}, indent=1))

    from vhh_rl.native_atom14.weighted_dpo import run_weighted_dpo

    summary = run_weighted_dpo(
        base_checkpoint=BASE,
        pairs_path=OUT / "pairs_train_mini.jsonl",
        pool_root=ROOT / "runs/native_pool",
        conditioning_dir=ROOT / "runs/native_pool/conditioning",
        weights_path=OUT / "residue_weights_mini.json",
        output_dir=OUT / "run",
        variant="cf",
        beta=10.0,
        lr=1e-5,
        max_steps=100,
        checkpoint_every=50,
        seed=20260913,
        log_tag="cf_dpo_mini",
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
