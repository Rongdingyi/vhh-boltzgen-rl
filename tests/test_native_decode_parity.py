"""Decode parity gate (task book §16): official writer sequence == re-decode."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

POOL = ROOT / "runs/native_pool"


def test_pool_decode_parity_recorded():
    """Every pool sample recorded decode_parity from the independent re-decode."""
    n = ok = 0
    for split in ("train", "heldout", "valid100"):
        for meta in (POOL / split).glob("*/metadata.jsonl"):
            for line in meta.open():
                row = json.loads(line)
                n += 1
                ok += bool(row.get("decode_parity"))
    assert n > 1500, f"unexpected pool size {n}"
    assert ok == n, f"decode parity failures: {n - ok}/{n}"
