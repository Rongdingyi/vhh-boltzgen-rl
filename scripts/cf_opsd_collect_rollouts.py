#!/usr/bin/env python
"""Phase A: collect OPSD rollouts (4 train + 4 heldout cases, K=8) + parity.

Parity: for the first train case, run the standard adapter.generate with the
same seed and require 100% identical sequences and endpoints.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml
import torch

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cf_opsd.rollout import collect_case_rollouts, save_trajectories  # noqa: E402
from vhh_rl.cli.common import case_spec_for  # noqa: E402
from vhh_rl.data.case import RLCase  # noqa: E402
from vhh_rl.native_atom14.adapter import NativeDesignAdapter  # noqa: E402
from vhh_rl.native_atom14.decode import fr_check  # noqa: E402
from vhh_rl.native_atom14.reward import make_reward_adapter  # noqa: E402

CONFIG = ROOT / "configs/cf_opsd/fixed_cases.yaml"
OUT = ROOT / "runs/cf_opsd/rollouts"
SEED_OFFSET = 800000


def load_cases() -> dict[str, RLCase]:
    cases = {}
    for line in (ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl").open():
        row = json.loads(line)
        cases[row["case_id"]] = RLCase(
            case_id=row["case_id"], structure_path=Path(row["structure_path"]),
            chain_id=row.get("chain_id", "A"), full_sequence=row["full_sequence"],
            design_positions=tuple(row["design_positions"]),
            fr_positions=tuple(row["fr_positions"]),
            split=row.get("split", "train"), seed_base=int(row.get("seed_base", 0)),
        )
    return cases


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    cases = load_cases()
    train = [cases[c] for c in cfg["train_cases"]]
    heldout = [cases[c] for c in cfg["heldout_cases"]]
    K = int(cfg["rollout"]["K"])
    steps = int(cfg["rollout"]["sampling_steps"])
    cond_steps = {max(0, min(steps - 1, int(round(p * steps)) - 1))
                  for p in cfg["query"]["candidate_progress"]} | {0}
    adapter = NativeDesignAdapter(device=0, sampling_steps=steps, diffusion_batch_size=K)
    reward = make_reward_adapter(cache_path=ROOT / "runs/cf_opsd/reward_cache.sqlite")

    # parity check on the first train case
    case = train[0]
    seed = case.seed_base + SEED_OFFSET
    spec = case.structure_path.parent / "design.yaml"
    shutil.rmtree(OUT / "_parity_ref" / case.case_id, ignore_errors=True)
    ref = adapter.generate(spec, case.case_id, OUT / "_parity_ref" / case.case_id,
                           num_designs=K, seed=seed)
    trajs, info = collect_case_rollouts(adapter, spec, case.case_id,
                                        OUT / "train" / case.case_id,
                                        num_designs=K, seed=seed,
                                        cond_steps=cond_steps)
    ref_seqs = [s.decoded_sequence for s in ref.samples]
    opsd_seqs = [t.endpoint_sequence for t in trajs]
    max_coord = max((ref.samples[i].coords - trajs[i].endpoint_coords).abs().max().item()
                    for i in range(K))
    parity = {
        "sequence_identical": ref_seqs == opsd_seqs,
        "max_endpoint_coord_diff": max_coord,
    }
    print(f"[parity] sequences identical={parity['sequence_identical']} "
          f"coord_maxdiff={max_coord:.2e}")
    assert parity["sequence_identical"], "sequence parity FAILED"
    assert max_coord < 1e-4, "endpoint coordinate parity FAILED"

    summary = {"parity": parity, "cases": {}}
    for split, group in (("train", train), ("heldout", heldout)):
        for case in group:
            seed = case.seed_base + SEED_OFFSET
            spec = case.structure_path.parent / "design.yaml"
            trajs, info = collect_case_rollouts(
                adapter, spec, case.case_id, OUT / split / case.case_id,
                num_designs=K, seed=seed, cond_steps=cond_steps)
            # score endpoints (valid, FR-clean only)
            seqs, idx = [], []
            for i, t in enumerate(trajs):
                fr = fr_check(t.endpoint_sequence, case.full_sequence, case.fr_positions)
                t.fr_mismatch = fr["fr_mismatch_count"]
                if not t.contains_invalid and t.fr_mismatch == 0:
                    seqs.append(t.endpoint_sequence)
                    idx.append(i)
            if seqs:
                batch = reward.score_sequences(seqs, spec=case_spec_for(case))
                for i, reward_value in zip(idx, batch.raw_scores.tolist()):
                    trajs[i].endpoint_reward = float(reward_value)
            n_reward = sum(1 for t in trajs if t.endpoint_reward is not None)
            save_trajectories(trajs, OUT / split / case.case_id / f"seed{seed}.pt",
                              contexts=info.get("contexts"),
                              cond_kwargs=info.get("cond_kwargs"),
                              cond_kwargs_steps=info.get("cond_kwargs_steps"))
            summary["cases"][f"{split}/{case.case_id}"] = {
                "n": len(trajs), "n_reward": n_reward, "seed": seed,
                "invalid": sum(1 for t in trajs if t.contains_invalid),
                "fr_mismatch": sum(1 for t in trajs if t.fr_mismatch > 0),
                "reward_mean": (sum(t.endpoint_reward for t in trajs
                                    if t.endpoint_reward is not None) / max(1, n_reward)),
                "elapsed_seconds": info["elapsed_seconds"],
            }
            print(f"[{split}] {case.case_id}: n={len(trajs)} reward_mean="
                  f"{summary['cases'][f'{split}/{case.case_id}']['reward_mean']:+.3f}")
    reward.close()
    (OUT / "collect_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {OUT / 'collect_summary.json'}")


if __name__ == "__main__":
    main()
