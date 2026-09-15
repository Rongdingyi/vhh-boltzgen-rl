"""Shared loaders for AG-CF-DPO experiments (no trainer changes)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

BASE_CKPT = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt")
POOL = ROOT / "runs/native_pool"
MANIFEST = ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl"
PAIRS = POOL / "pairs_train.jsonl"
RESIDUE_CREDIT = ROOT / "runs/next_stage/counterfactual/residue_credit.csv"
REGION_CREDIT = ROOT / "runs/next_stage/counterfactual/region_credit.csv"
CURRENT_WEIGHTS = ROOT / "runs/next_stage/weights/residue_weights.json"
AG_ROOT = ROOT / "runs/adaptive_granularity"
WEIGHTS_DIR = AG_ROOT / "weights"
AUDIT_DIR = AG_ROOT / "audit"
PILOT_DIR = AG_ROOT / "pilot"
FULL_DIR = AG_ROOT / "full"

# §29: fixed pilot protocol (never changed, not selected post-hoc)
PILOT_TRAIN_CASES = ["sab2_6u52_c", "sab2_7sl5_d", "sab2_7nqk_b", "sab2_6mqe_h"]
HELDOUT4 = ["sab2_4hf5_h", "sab2_4mwf_h", "sab2_5mp6_h", "sab2_6cvk_b2"]

VARIANTS = ["no_floor", "strict_consensus", "strict_region",
            "adaptive", "adaptive_shuffle"]
VARIANT_LABEL = {"current_cf": "Current CF", "no_floor": "No-floor",
                 "strict_consensus": "Strict consensus",
                 "strict_region": "Strict region", "adaptive": "Adaptive",
                 "adaptive_shuffle": "Adaptive shuffle",
                 "adaptive_eligible_cf": "Eligible-CF"}


def load_pairs() -> list[dict]:
    return [json.loads(l) for l in PAIRS.open()]


def pair_id(pair: dict) -> str:
    return f"{pair['case_id']}:{pair['winner_sample_id']}:{pair['loser_sample_id']}"


def manifest_splits() -> dict[str, str]:
    return {json.loads(l)["case_id"]: json.loads(l)["split"] for l in MANIFEST.open()}


def heldout8() -> list[str]:
    return sorted(c for c, s in manifest_splits().items() if s == "test")


def _f(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in (None, "", "nan"):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def load_residue_rows() -> dict[str, list[dict]]:
    """pair_id -> [{position, region, c_drop, c_gain, c_cons, ...}]"""
    out: dict[str, list[dict]] = {}
    for row in csv.DictReader(RESIDUE_CREDIT.open()):
        row["position"] = int(row["position"])
        for key in ("c_drop", "c_gain", "c_avg", "c_cons"):
            row[key] = _f(row, key)
        out.setdefault(row["pair_id"], []).append(row)
    for rows in out.values():
        rows.sort(key=lambda r: r["position"])
    return out


def load_region_rows() -> dict[str, dict[str, dict]]:
    """pair_id -> region -> {G_drop, G_gain, G_cons, G_avg, epistasis_*, ...}"""
    out: dict[str, dict[str, dict]] = {}
    for row in csv.DictReader(REGION_CREDIT.open()):
        for key in ("G_drop", "G_gain", "G_cons", "G_avg",
                    "epistasis_avg", "epistasis_cons"):
            row[key] = _f(row, key)
        row["n_changed_positions"] = int(row["n_changed_positions"])
        if row.get("G_cons") is None and row["G_drop"] is not None \
                and row["G_gain"] is not None:
            row["G_cons"] = min(row["G_drop"], row["G_gain"])
        out.setdefault(row["pair_id"], {})[row["region"]] = row
    return out


def load_current_weights() -> dict:
    return json.loads(CURRENT_WEIGHTS.read_text())


def load_cf_pair_weights(pid: str) -> dict[int, float]:
    payload = load_current_weights()
    entry = payload["pairs"].get(pid)
    if entry is None:
        return {}
    return {int(k): float(v) for k, v in entry["cf"].items()}


def load_ag_weights(path: Path | None = None) -> dict:
    path = path or (WEIGHTS_DIR / "ag_weights.json")
    return json.loads(path.read_text())


def sha256(path: Path) -> str | None:
    import hashlib
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ag_pairs_file(variant: str, pilot: bool = False) -> list[dict]:
    suffix = "_pilot4" if pilot else ""
    path = WEIGHTS_DIR / f"pairs_{variant}{suffix}.jsonl"
    return [json.loads(l) for l in path.open()]
