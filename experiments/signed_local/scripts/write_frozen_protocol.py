#!/usr/bin/env python
"""Write experiments/signed_local/configs/FROZEN_PROTOCOL.yaml (task book §98)."""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path
import yaml

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
OUT = ROOT / "experiments/signed_local/configs/FROZEN_PROTOCOL.yaml"


def sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    d = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def main() -> None:
    try:
        repo_commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                     capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        repo_commit = None
    boltzgen_commit = subprocess.run(
        ["git", "-C", "/share/home/rongdingyi/programs/proteingen/boltzgen", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    payload = {
        "version": 1,
        "method": "sl_cf_dpo",
        "repo_commit": repo_commit,
        "boltzgen_commit": boltzgen_commit,
        "base_checkpoint_sha256": sha(Path(
            "/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt")),
        "pairs_train_sha256": sha(ROOT / "runs/native_pool/pairs_train.jsonl"),
        "residue_credit_sha256": sha(ROOT / "runs/next_stage/counterfactual/residue_credit.csv"),
        "counterfactual_scores_sha256": sha(
            ROOT / "runs/next_stage/counterfactual/counterfactual_scores.jsonl"),
        "residue_weights_sha256": sha(ROOT / "runs/next_stage/weights/residue_weights.json"),
        "train_edges_sha256": sha(ROOT / "runs/signed_local/edges/train_edges.jsonl"),
        "heldout_edges_sha256": sha(ROOT / "runs/signed_local/edges/heldout_edges.jsonl"),
        "train_cases": ["sab2_6u52_c", "sab2_7sl5_d", "sab2_7nqk_b", "sab2_6mqe_h"],
        "heldout_cases": ["sab2_4hf5_h", "sab2_4mwf_h", "sab2_5mp6_h", "sab2_6cvk_b2"],
        "reward_tolerance": 0.05,
        "beta_global": 10.0,
        "beta_local": 10.0,
        "eta": 0.75,
        "lr": 1.0e-5,
        "schedule": "deterministic_3_to_1",
        "eval_seeds": "seed_base + 800000 (8 samples/case)",
    }
    OUT.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
