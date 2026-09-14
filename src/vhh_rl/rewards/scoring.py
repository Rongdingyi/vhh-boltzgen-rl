"""Reward adapters: ToyReward (§25) and the black-box VHH scorer (§7).

The real scorer runs in a separate process (persistent JSONL worker) because
it lives in the vhh-guidance-esmc env while the policy needs the
vhh-guidance/BoltzGen env.  Scores are cached by sha1(sequence).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ScoreBatch:
    raw_scores: torch.Tensor
    optimization_scores: torch.Tensor
    info: dict = None  # optional per-batch diagnostics (e.g. guardrail stats)

    def __post_init__(self):
        if self.info is None:
            self.info = {}


class ScorerAdapter:
    """Black-box sequence scorer.  Larger optimization score = better."""

    def __init__(
        self,
        worker_python: str,
        worker_script: str,
        scorer_root: str,
        objective: str = "native_probability",
        cache_path: str | Path | None = None,
        case_spec: dict | None = None,
        guard_lambda: float = 0.0,
    ) -> None:
        self.worker_python = worker_python
        self.worker_script = worker_script
        self.scorer_root = scorer_root
        self.objective = objective
        self.case_spec = case_spec or {}
        self.guard_lambda = float(guard_lambda)
        # Cache is namespaced per objective: raw scores from one objective must
        # never be read as another objective's (round-1 nat_prob vs round-2
        # margin have different scales and meanings).
        import re as _re
        self._table = ("scores" if objective == "native_probability"
                       else "scores_" + _re.sub(r"[^a-z0-9]+", "_", objective.lower()).strip("_"))
        self.cache_path = Path(cache_path) if cache_path else None
        self._proc: subprocess.Popen | None = None
        self._request_id = 0
        self._conn: sqlite3.Connection | None = None
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.cache_path))
            self._conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table} (sha1 TEXT PRIMARY KEY, raw REAL)"
            )
            cols = [r[1] for r in self._conn.execute(f"PRAGMA table_info({self._table})")]
            if "guard" not in cols:
                self._conn.execute(f"ALTER TABLE {self._table} ADD COLUMN guard REAL")
            self._conn.commit()

    # ------------------------------------------------------------------ cache

    def _cached(self, hashes: list[str]) -> dict[str, tuple[float, float | None]]:
        out: dict[str, tuple[float, float | None]] = {}
        if self._conn is not None:
            for h in hashes:
                row = self._conn.execute(
                    f"SELECT raw, guard FROM {self._table} WHERE sha1=?", (h,)
                ).fetchone()
                if row is not None:
                    out[h] = (float(row[0]), None if row[1] is None else float(row[1]))
        return out

    def _store(self, hashes: list[str], raws: list[float],
               guards: list[float | None] | None = None) -> None:
        if self._conn is not None:
            if guards is None:
                guards = [None] * len(hashes)
            self._conn.executemany(
                f"INSERT OR REPLACE INTO {self._table} (sha1, raw, guard) VALUES (?, ?, ?)",
                list(zip(hashes, raws, guards)),
            )
            self._conn.commit()

    # ----------------------------------------------------------------- worker

    def _ensure_worker(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        self._proc = subprocess.Popen(
            [
                self.worker_python,
                "-u",
                self.worker_script,
                "--scorer-root",
                self.scorer_root,
                "--objective",
                self.objective,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return self._proc

    def score_sequences(self, sequences: list[str], spec: dict | None = None) -> ScoreBatch:
        hashes = [hashlib.sha1(s.encode()).hexdigest() for s in sequences]
        cached = self._cached(hashes)
        pending = [(i, h, s) for i, (h, s) in enumerate(zip(hashes, sequences)) if h not in cached]
        raws = [0.0] * len(sequences)
        guards: list[float | None] = [None] * len(sequences)
        for i, h in enumerate(hashes):
            if h in cached:
                raws[i], guards[i] = cached[h]
        if pending:
            request_spec = spec if spec is not None else self.case_spec
            reply = self._score_request(request_spec, [p[2] for p in pending])
            raw_values = [float(v) for v in reply["scores"]]
            guard_values = ([float(v) for v in reply["guard"]]
                            if "guard" in reply else [None] * len(pending))
            for (idx, h, _s), value, g in zip(pending, raw_values, guard_values):
                raws[idx] = float(value)
                guards[idx] = g
            self._store([p[1] for p in pending], raw_values, guard_values)
        tensor = torch.tensor(raws, dtype=torch.float32)
        opt = tensor if self.direction_is_higher() else -tensor
        info: dict = {}
        # Round-2 guardrail: opt = main - lambda * max(0, floor - nativeness).
        # The floor rides in the per-request spec (injected by _PerCaseScorer
        # from the floors file); absent floor => guardrail disabled.  Cached
        # rows without a guard value are treated as non-violating.
        floor = (spec or self.case_spec).get("guard_floor", None)
        if floor is not None and any(g is not None for g in guards):
            nat = torch.tensor([g if g is not None else float(floor) for g in guards])
            viol = torch.clamp(float(floor) - nat, min=0.0)
            opt = tensor - self.guard_lambda * viol
            info = {"guard_nat_mean": float(nat.mean()),
                    "guard_viol_rate": float((viol > 0).float().mean()),
                    "guard_floor": float(floor)}
        return ScoreBatch(raw_scores=tensor, optimization_scores=opt, info=info)

    def _score_request(self, request_spec: dict, seqs: list[str]) -> dict:
        """Send one scoring request; respawn the worker once if it died."""
        import time
        for attempt in (0, 1):
            worker = self._ensure_worker()
            assert worker.stdin is not None and worker.stdout is not None
            try:
                worker.stdin.write(
                    json.dumps({"spec": request_spec, "sequences": seqs}) + "\n"
                )
                worker.stdin.flush()
                reply = json.loads(worker.stdout.readline())
                return reply
            except (json.JSONDecodeError, BrokenPipeError, OSError):
                if attempt == 1:
                    raise
                time.sleep(5)
                self.close()
        raise RuntimeError("unreachable")

    def direction_is_higher(self) -> bool:
        return True  # AUDIT: vhh_guidance native_probability is a probability; higher = more camelid-native

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.wait(timeout=60)
        self._proc = None
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class ToySequenceReward:
    """Fraction of CDR positions equal to one target AA (task book §25).

    The target AA is chosen per case as the least-frequent AA that is allowed
    at every design position of the baseline rollouts.
    """

    def __init__(self, target_aa: str) -> None:
        if target_aa not in "ACDEFGHIKLMNPQRSTVWY":
            raise ValueError(target_aa)
        self.target = target_aa

    @staticmethod
    def pick_target(sequences: list[str], design_positions: tuple[int, ...]) -> str:
        counts = {aa: 0 for aa in "ACDEFGHIKLMNPQRSTVWY"}
        n = 0
        for seq in sequences:
            for pos in design_positions:
                aa = seq[pos]
                if aa in counts:
                    counts[aa] += 1
                    n += 1
        if n == 0:
            raise ValueError("no design positions counted")
        # Only AAs the (constraint-masked) baseline actually samples are valid
        # targets: the nanobody protocol forbids Cys via aa_constraint_mask, so
        # a globally-minimal but forbidden AA would make the toy reward
        # unreachable and the policy gradient would correctly stay at zero.
        reachable = {a: c for a, c in counts.items() if c > 0}
        return min(reachable, key=lambda a: reachable[a])

    def score_sequences(self, sequences: list[str], design_positions: tuple[int, ...]) -> ScoreBatch:
        raw = torch.tensor(
            [
                sum(1 for p in design_positions if s[p] == self.target) / max(1, len(design_positions))
                for s in sequences
            ],
            dtype=torch.float32,
        )
        return ScoreBatch(raw_scores=raw, optimization_scores=raw)

    def close(self) -> None:  # interface parity with ScorerAdapter
        return None
