"""Held-out evaluation for CF-OPSD / CF-DPO-mini / base (task book §50)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ..data.case import RLCase
from ..cli.common import case_spec_for
from ..native_atom14.adapter import NativeDesignAdapter
from ..native_atom14.decode import fr_check
from ..native_atom14.reward import make_reward_adapter

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")


def load_cases(case_ids: list[str], manifest: Path | None = None) -> dict[str, RLCase]:
    wanted = set(case_ids)
    out = {}
    manifest = Path(manifest) if manifest else (ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl")
    for line in manifest.open():
        row = json.loads(line)
        if row["case_id"] not in wanted:
            continue
        out[row["case_id"]] = RLCase(
            case_id=row["case_id"], structure_path=Path(row["structure_path"]),
            chain_id=row.get("chain_id", "A"), full_sequence=row["full_sequence"],
            design_positions=tuple(row["design_positions"]),
            fr_positions=tuple(row["fr_positions"]),
            split=row.get("split", "train"), seed_base=int(row.get("seed_base", 0)),
        )
    return out


def evaluate(case_ids: list[str], ckpt: Path | None, tag: str, *,
             num_samples: int = 8, sampling_steps: int = 50,
             seed_offset: int = 800000, run_root: Path | None = None,
             manifest: Path | None = None) -> dict:
    cases = load_cases(case_ids, manifest=manifest)
    kwargs = dict(sampling_steps=sampling_steps, diffusion_batch_size=num_samples)
    if ckpt is not None:
        kwargs["design_ckpt"] = Path(ckpt)
    adapter = NativeDesignAdapter(**kwargs)
    reward = make_reward_adapter(cache_path=ROOT / "runs/cf_opsd/reward_cache.sqlite")
    run_root = run_root or (ROOT / "runs/cf_opsd/eval" / tag)
    out = {"tag": tag, "checkpoint": str(ckpt) if ckpt else "native_base", "cases": {}}
    for cid in sorted(cases):
        case = cases[cid]
        capture = adapter.generate(case.structure_path.parent / "design.yaml", cid,
                                   run_root / cid, num_designs=num_samples,
                                   seed=case.seed_base + seed_offset)
        seqs, rewards, n_invalid, n_fr = [], [], 0, 0
        for s in capture.samples:
            fr = fr_check(s.decoded_sequence, case.full_sequence, case.fr_positions)
            n_invalid += int(s.contains_invalid)
            n_fr += int(fr["fr_mismatch_count"] > 0)
            if not s.contains_invalid and fr["fr_mismatch_count"] == 0:
                seqs.append(s.decoded_sequence)
        if seqs:
            batch = reward.score_sequences(seqs, spec=case_spec_for(case))
            rewards = [float(v) for v in batch.raw_scores.tolist()]
        mean = sum(rewards) / len(rewards) if rewards else None
        out["cases"][cid] = {"n": len(capture.samples), "n_scored": len(rewards),
                             "n_invalid": n_invalid, "n_fr_mismatch": n_fr,
                             "n_unique": len(set(seqs)),
                             "reward_mean": mean}
        print(f"[{tag}] {cid}: reward={mean}", flush=True)
    reward.close()
    rewards_all = [v["reward_mean"] for v in out["cases"].values() if v["reward_mean"] is not None]
    out["reward_mean"] = sum(rewards_all) / len(rewards_all) if rewards_all else None
    (run_root / "eval_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (run_root / "eval_summary.json").write_text(json.dumps(out, indent=1))
    return out
