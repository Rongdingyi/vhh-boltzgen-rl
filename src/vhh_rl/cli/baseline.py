"""Gate 0 baseline: original BoltzGen IF, K sequences per case."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ..adapters.boltzgen_if import BoltzGenIFAdapter
from ..config import RLConfig
from ..rewards.scoring import ScorerAdapter, ToySequenceReward
from .common import build_adapter, case_spec_for, capture_cases, ensure_slurm, log_factory, manifest_cases


def run_baseline(cfg: RLConfig, output_dir: Path) -> int:
    ensure_slurm()
    log = log_factory(output_dir)
    cases = manifest_cases(cfg, splits=("train", "val", "test"))
    if cfg.max_cases:
        cases = cases[: cfg.max_cases]
    adapter = build_adapter(cfg)
    captured = capture_cases(adapter, cfg, cases, output_dir / "captures")
    scorer = _build_scorer(cfg, cases)
    rows = []
    summary_cases = []
    for case, cap in zip(cases, captured):
        seqs = []
        for k in range(cfg.sampling.group_size):
            traj = adapter.rollout_with_logprobs(
                cap, _temperature(cfg), seed=case.seed_base * 1000 + k
            )
            if not adapter.fr_unchanged(cap, traj, case.full_sequence):
                raise RuntimeError(f"baseline FR mutation in {case.case_id}")
            seqs.append(traj.final_sequence)
        if isinstance(scorer, ToySequenceReward):
            score = scorer.score_sequences(seqs, cap.design_positions)
        else:
            score = scorer.score_sequences(seqs, case)
        raw = score.raw_scores.tolist()
        for k, (seq, value) in enumerate(zip(seqs, raw)):
            rows.append(
                {
                    "case_id": case.case_id,
                    "seed": case.seed_base * 1000 + k,
                    "sequence": seq,
                    "raw_score": value,
                    "optimization_score": float(score.optimization_scores[k]),
                    "temperature": _temperature(cfg),
                }
            )
        summary_cases.append(
            {
                "case_id": case.case_id,
                "mean": sum(raw) / len(raw),
                "median": sorted(raw)[len(raw) // 2],
                "best_of_n": max(raw),
            }
        )
        log(f"[baseline] {case.case_id}: mean={summary_cases[-1]['mean']:.4f} best={max(raw):.4f}")
    scorer.close()
    with (output_dir / "baseline_sequences.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "baseline_sequences.fasta").write_text(
        "".join(f">{r['case_id']}_{r['seed']}\n{r['sequence']}\n" for r in rows)
    )
    (output_dir / "baseline_summary.json").write_text(
        json.dumps({"cases": summary_cases}, indent=2)
    )
    return 0


def _temperature(cfg: RLConfig) -> float:
    if cfg.sampling.temperature == "auto":
        return 0.5  # placeholder; scripts/temperature_sweep.py picks the real one
    return float(cfg.sampling.temperature)


def _build_scorer(cfg: RLConfig, cases):
    if cfg.reward.name == "toy":
        return ToySequenceReward(target_aa="W")
    specs = {case.case_id: case_spec_for(case) for case in cases}
    # one adapter per case so the worker receives the right CDR regions
    return _PerCaseScorer(cfg, specs)


class _PerCaseScorer:
    """One shared worker process for all cases.

    The worker loads the full guidance ensemble once (each extra worker holds
    another complete copy on the GPU and OOMs on multi-case runs); the per-case
    spec is passed per request instead (the worker protocol already accepts a
    spec per message).
    """

    def __init__(self, cfg: RLConfig, specs: dict[str, dict]) -> None:
        self.cfg = cfg
        self.specs = specs
        self._adapter: ScorerAdapter | None = None
        self.floors: dict[str, float] = {}
        if cfg.reward.guard_floors_path:
            import json as _json
            self.floors = {k: float(v) for k, v in
                           _json.loads(Path(cfg.reward.guard_floors_path).read_text()).items()}

    def _get_adapter(self) -> ScorerAdapter:
        if self._adapter is None:
            self._adapter = ScorerAdapter(
                worker_python="/share/home/rongdingyi/.conda/envs/vhh-guidance-esmc/bin/python",
                worker_script=str(
                    Path(__file__).resolve().parents[3]
                    / "src/vhh_rl/rewards/scorer_worker.py"
                ),
                scorer_root=self.cfg.scorer_root,
                objective=self.cfg.reward.objective,
                cache_path=Path(self.cfg.output_root) / "scorer_cache.sqlite3",
                case_spec=next(iter(self.specs.values())),
                guard_lambda=self.cfg.reward.guard_lambda,
            )
        return self._adapter

    def score_sequences(self, sequences, case) -> object:
        spec = dict(self.specs[case.case_id])
        if case.case_id in self.floors:
            spec["guard_floor"] = self.floors[case.case_id]
        return self._get_adapter().score_sequences(sequences, spec=spec)

    def close(self) -> None:
        if self._adapter is not None:
            self._adapter.close()
