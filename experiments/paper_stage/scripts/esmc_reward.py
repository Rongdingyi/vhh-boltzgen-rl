#!/usr/bin/env python
"""Second black-box reward: VHH-ESM-C exact masked CDR PLL (paper stage §37-§41).

Central scorer with a sqlite cache keyed by sha1(case_id|sequence).

usage (CLI):
  esmc_reward.py --in records.jsonl --out scored.jsonl
records.jsonl rows: {"key": str, "case_id": str, "sequence": str}
output rows add: cdr_pll, full_pll, status
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import torch

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
GUID = Path("/share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance")
ESM = Path("/share/home/rongdingyi/programs/proteingen/esm")
for p in (str(GUID / "src"), str(ESM), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_MODEL = "/share/data/limc/esmc-600m-vhh"


class ESMCReward:
    def __init__(self, cache_path: str | Path | None = None, device: str | None = None,
                 model: str = DEFAULT_MODEL, dtype=torch.bfloat16) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.dtype = dtype
        self._adapter = None
        self.conn = None
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(cache_path))
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS esmc (key TEXT PRIMARY KEY, cdr_pll REAL, full_pll REAL)")
            self.conn.commit()

    @property
    def adapter(self):
        if self._adapter is None:
            from vhh_esmc_guidance.adapters.vhh_esmc import VHHESMCAdapter

            self._adapter = VHHESMCAdapter.from_pretrained(
                self.model, device=self.device, torch_dtype=self.dtype)
        return self._adapter

    @staticmethod
    def key(case_id: str, sequence: str) -> str:
        return hashlib.sha1(f"{case_id}|{sequence}".encode()).hexdigest()

    def _cached(self, keys: list[str]) -> dict[str, tuple[float, float]]:
        out = {}
        if self.conn is None:
            return out
        for k in keys:
            row = self.conn.execute(
                "SELECT cdr_pll, full_pll FROM esmc WHERE key=?", (k,)).fetchone()
            if row is not None:
                out[k] = (float(row[0]), float(row[1]))
        return out

    def score(self, records: list[dict], cases: dict[str, dict], batch_size: int = 32) -> list[dict]:
        """records: [{key, case_id, sequence}]; cases: {case_id: {native_sequence, design_positions}}"""
        keys = [self.key(r["case_id"], r["sequence"]) for r in records]
        cached = self._cached(keys)
        pending = [(i, r, k) for i, (r, k) in enumerate(zip(records, keys)) if k not in cached]
        out = [dict(r) for r in records]
        for i, r in enumerate(records):
            k = keys[i]
            if k in cached:
                out[i]["cdr_pll"], out[i]["full_pll"] = cached[k]
                out[i]["status"] = "PASS"
        if pending:
            from vhh_esmc_guidance.eval.sequence_report import score_candidates

            scored = score_candidates(self.adapter, [r for _i, r, _k in pending], cases,
                                      batch_size=batch_size)
            store = []
            for (i, _r, k), s in zip(pending, scored):
                if s.get("status") == "PASS" and s.get("cdr_pll") is not None:
                    out[i]["cdr_pll"] = float(s["cdr_pll"])
                    out[i]["full_pll"] = float(s["full_pll"])
                    out[i]["status"] = "PASS"
                    store.append((k, out[i]["cdr_pll"], out[i]["full_pll"]))
                else:
                    out[i]["status"] = s.get("status", "INVALID")
                    out[i]["cdr_pll"] = None
                    out[i]["full_pll"] = None
            if self.conn is not None and store:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO esmc (key, cdr_pll, full_pll) VALUES (?,?,?)", store)
                self.conn.commit()
        return out


def _case_entry(row: dict) -> dict:
    from vhh_rl.data.cdr import cdr_range

    class _C:
        design_positions = tuple(int(p) for p in row["design_positions"])
        full_sequence = row["full_sequence"]

    return {
        "native_sequence": row["full_sequence"],
        "design_positions": [int(p) for p in row["design_positions"]],
        "cdr_groups": [list(cdr_range(_C(), k)) for k in ("cdr1", "cdr2", "cdr3")],
    }


def load_cases(split: str = "train") -> dict[str, dict]:
    """split: "train" | "test" | "all" (includes valid100 always)."""
    wanted = {"train"} if split == "train" else {"test"} if split == "test" else {"train", "test"}
    cases = {}
    for line in (ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl").open():
        row = json.loads(line)
        if row.get("split") not in wanted:
            continue
        cases[row["case_id"]] = _case_entry(row)
    for line in (ROOT / "runs/round1_rl_split/rl_manifest_valid100test.jsonl").open():
        row = json.loads(line)
        cases[row["case_id"]] = _case_entry(row)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path,
                        default=ROOT / "runs/paper_stage/second_reward/esmc_cache.sqlite")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    records = [json.loads(l) for l in args.inp.open()]
    cases = load_cases("all")
    scorer = ESMCReward(cache_path=args.cache)
    scored = scorer.score(records, cases, batch_size=args.batch_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for row in scored:
            fh.write(json.dumps(row) + "\n")
    n_ok = sum(1 for r in scored if r.get("status") == "PASS")
    print(f"scored {n_ok}/{len(scored)} -> {args.out}")


if __name__ == "__main__":
    main()
