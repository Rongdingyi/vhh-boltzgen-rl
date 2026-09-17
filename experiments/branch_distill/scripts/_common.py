"""Shared helpers for the branch_distill experiment scripts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

BASE_CKPT = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt")
MANIFEST = ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl"
CONFIG_DIR = ROOT / "experiments/branch_distill/configs"
FROZEN = CONFIG_DIR / "FROZEN_PROTOCOL.yaml"
RUN_ROOT = ROOT / "runs/branch_distill"
GATE1_DIR = RUN_ROOT / "gate1"
GATE2_DIR = RUN_ROOT / "gate2"
GATE3_DIR = RUN_ROOT / "gate3"
COMMON_ROUND1 = GATE3_DIR / "common_round1"
DOCS = ROOT / "docs/branch_distill"
DEVICE = "cuda"

TRAIN_CASES = ["sab2_6u52_c", "sab2_7sl5_d", "sab2_7nqk_b", "sab2_6mqe_h"]
HELDOUT_CASES = ["sab2_4hf5_h", "sab2_4mwf_h", "sab2_5mp6_h", "sab2_6cvk_b2"]
K_SIBLINGS = 8
SAMPLING_STEPS = 50
CANDIDATE_PROGRESS = [0.60, 0.70, 0.80, 0.90]
EVAL_SEED_OFFSET = 700000


def load_cases(case_ids=None) -> dict:
    wanted = set(case_ids or (TRAIN_CASES + HELDOUT_CASES))
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


def protocol_hash() -> str:
    from vhh_rl.branch_distill.gates import protocol_hash as ph
    return ph(FROZEN)


def sha256(path: Path) -> str | None:
    from vhh_rl.branch_distill.gates import protocol_hash as ph
    return ph(path) if Path(path).is_file() else None


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))


def make_adapter(ckpt: Path, *, num_designs: int = 2,
                 sampling_steps: int = SAMPLING_STEPS, batch: int = K_SIBLINGS):
    from vhh_rl.native_atom14.adapter import NativeDesignAdapter
    adapter = NativeDesignAdapter(design_ckpt=Path(ckpt), sampling_steps=sampling_steps,
                                  diffusion_batch_size=batch)
    adapter.num_designs = int(num_designs)
    return adapter


def make_scorer():
    from vhh_rl.native_atom14.reward import make_reward_adapter
    return make_reward_adapter(cache_path=ROOT / "runs/cf_opsd/reward_cache.sqlite")


def case_spec(case):
    from vhh_rl.cli.common import case_spec_for
    return case_spec_for(case)


def require_gate(path: Path, key: str, expected, override: str | None, label: str) -> dict:
    if path.is_file():
        payload = json.loads(path.read_text())
        if payload.get(key) is True or payload.get("verdict") == expected:
            return payload
    if override:
        print(f"[warn] {label} override: {override}", flush=True)
        return {"overridden": True, "reason": override}
    raise SystemExit(f"{label} gate not passed ({path}); use --override-gate REASON for debug only")


def guard_protocol(artifact: dict) -> None:
    from vhh_rl.branch_distill.gates import guard_protocol as guard
    guard(artifact, FROZEN)
