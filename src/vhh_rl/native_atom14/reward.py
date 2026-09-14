"""Reward adapter for native decoded sequences (task book §2/§5).

The ONLY task reward is the existing CDR classifier reward.  This module
reuses the round-1 scorer worker (`vhh_rl/rewards/scorer_worker.py`) and the
round-1 ScorerAdapter; the optimized value is `cdr_camelid_margin`
(objective `cdr_margin_guarded`, direction: higher is better).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vhh_rl.rewards.scoring import ScorerAdapter  # noqa: E402
from vhh_rl.cli.common import case_spec_for  # noqa: E402

WORKER_PYTHON = "/share/home/rongdingyi/.conda/envs/vhh-guidance-esmc/bin/python"
WORKER_SCRIPT = str(ROOT / "vhh_rl/rewards/scorer_worker.py")
DEFAULT_SCORER_ROOT = "/share/home/rongdingyi/programs/proteingen/vhh_guidance"


def make_reward_adapter(
    cache_path: str | Path | None = None,
    scorer_root: str = DEFAULT_SCORER_ROOT,
    objective: str = "cdr_margin_guarded",
) -> ScorerAdapter:
    return ScorerAdapter(
        worker_python=WORKER_PYTHON,
        worker_script=WORKER_SCRIPT,
        scorer_root=scorer_root,
        objective=objective,
        cache_path=Path(cache_path) if cache_path else None,
    )


def score_case_sequences(adapter: ScorerAdapter, case, sequences: list[str]):
    """Score full sequences with the case's CDR spec. Returns ScoreBatch."""
    spec = case_spec_for(case)
    return adapter.score_sequences(sequences, spec=spec)
