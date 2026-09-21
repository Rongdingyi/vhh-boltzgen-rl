"""Shared helpers for online_pref experiment scripts."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

BASE_CKPT = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt")
MANIFEST = ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl"
CONFIG_DIR = ROOT / "experiments/online_pref/configs"
FROZEN = CONFIG_DIR / "FROZEN_PROTOCOL.yaml"
RUN_ROOT = ROOT / "runs/online_pref"
PHASE0 = RUN_ROOT / "phase0"
EVAL = PHASE0 / "eval"
DOCS = ROOT / "docs/online_pref"
RESULTS = ROOT / "results/online_pref"
DEVICE = "cuda"

TRAIN_CASES = ["sab2_6u52_c", "sab2_7sl5_d", "sab2_7nqk_b", "sab2_6mqe_h"]
TRAIN_SEEDS = [20260915, 43, 44]
K_SIBLINGS = 8
SAMPLING_STEPS = 50
BRANCH_PROGRESS = 0.60
EVAL_SEED_OFFSET = 700000
MIN_REWARD_GAP = 0.30
EMA_DECAY = 0.99
ROUNDS = 4
UPDATES_PER_ROUND = 25


def heldout8() -> list[str]:
    out = []
    for line in MANIFEST.open():
        row = json.loads(line)
        if row.get("split") == "test":
            out.append(row["case_id"])
    return sorted(out)


def load_cases(case_ids=None) -> dict:
    wanted = set(case_ids or (TRAIN_CASES + heldout8()))
    out = {}
    for line in MANIFEST.open():
        row = json.loads(line)
        if row["case_id"] not in wanted:
            continue
        from vhh_rl.data.case import RLCase
        out[row["case_id"]] = RLCase(
            case_id=row["case_id"], structure_path=Path(row["structure_path"]),
            chain_id=row.get("chain_id", "A"), full_sequence=row["full_sequence"],
            design_positions=tuple(row["design_positions"]),
            fr_positions=tuple(row["fr_positions"]),
            split=row.get("split", "train"), seed_base=int(row.get("seed_base", 0)))
    return out


def spec_path(case) -> Path:
    return case.structure_path.parent / "design.yaml"


def progress_to_step(progress: float, n_steps: int = SAMPLING_STEPS) -> int:
    return max(0, min(n_steps - 1, int(round(progress * n_steps)) - 1))


def sha256(path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_hash() -> str:
    from vhh_rl.online_pref.gates import protocol_hash as ph
    return ph(FROZEN)


def write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=str))


def make_scorer():
    from vhh_rl.native_atom14.reward import make_reward_adapter
    return make_reward_adapter(cache_path=ROOT / "runs/cf_opsd/reward_cache.sqlite")


def case_spec(case):
    from vhh_rl.cli.common import case_spec_for
    return case_spec_for(case)


def seed_dir(seed: int) -> Path:
    return PHASE0 / f"seed_{seed}"
