"""Gate 6 evaluation: RL IF vs original IF on held-out cases (§45/§56)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from ..adapters.boltzgen_if import BoltzGenIFAdapter
from ..config import RLConfig
from .baseline import _PerCaseScorer
from .common import build_adapter, case_spec_for, capture_cases, ensure_slurm, log_factory, manifest_cases


def run_evaluate(cfg: RLConfig, output_dir: Path) -> int:
    ensure_slurm()
    log = log_factory(output_dir)
    cases = manifest_cases(cfg, splits=("test",))
    if cfg.max_cases:
        cases = cases[: cfg.max_cases]
    adapter = build_adapter(cfg)
    captured = capture_cases(adapter, cfg, cases, output_dir / "captures")
    import os as _os
    ckpt_env = _os.environ.get("VHHRL_EVAL_CKPT", "")
    if ckpt_env:
        checkpoint = Path(ckpt_env)
    else:
        checkpoint = output_dir.parent / "scorer_overfit" / "checkpoints" / "checkpoint_0004.pt"
    if checkpoint.is_file():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        adapter.decoder.load_state_dict(payload["decoder_state_dict"])
        log(f"[eval] loaded RL checkpoint {checkpoint}")
    else:
        log("[eval] no RL checkpoint found; evaluating base policy only")

    scorer = _PerCaseScorer(cfg, {c.case_id: case_spec_for(c) for c in cases})
    rows = []
    for case, cap in zip(cases, captured):
        base_seqs, rl_seqs = [], []
        for k in range(cfg.evaluation.num_samples_per_case):
            seed = case.seed_base * 1000 + k
            base_seqs.append(adapter.rollout_with_logprobs(cap, 0.5, seed=seed).final_sequence)
            rl_seqs.append(adapter.rollout_with_logprobs(cap, 0.5, seed=seed + 500000).final_sequence)
        base_scores = scorer.score_sequences(base_seqs, case).raw_scores.tolist()
        rl_scores = scorer.score_sequences(rl_seqs, case).raw_scores.tolist()
        for k in range(len(base_seqs)):
            rows.append(
                {
                    "case_id": case.case_id,
                    "arm": "base",
                    "seed": seed,
                    "sequence": base_seqs[k],
                    "raw_score": base_scores[k],
                }
            )
            rows.append(
                {
                    "case_id": case.case_id,
                    "arm": "rl",
                    "seed": seed,
                    "sequence": rl_seqs[k],
                    "raw_score": rl_scores[k],
                }
            )
        log(
            f"[eval] {case.case_id}: base mean {sum(base_scores)/len(base_scores):.4f} "
            f"rl mean {sum(rl_scores)/len(rl_scores):.4f}"
        )
    scorer.close()
    with (output_dir / "eval_per_case.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = _summarize(rows)
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


def _summarize(rows: list[dict]) -> dict:
    by_case: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        by_case.setdefault(r["case_id"], {}).setdefault(r["arm"], []).append(float(r["raw_score"]))
    summary: dict[str, dict] = {}
    for case_id, arms in by_case.items():
        summary[case_id] = {
            arm: {
                "mean": sum(v) / len(v),
                "median": sorted(v)[len(v) // 2],
                "best": max(v),
            }
            for arm, v in arms.items()
        }
    return summary
