#!/usr/bin/env python
"""Phase 1: temporal arms on the frozen Phase 0 B pair banks (§22-§24).

T-full / T-suffix / T-prefix read the exact same banks and update schedules as
Phase 0 B; only the sigma domain differs, driven by the same uniform seed.
"""
from __future__ import annotations

import argparse
import json

import _common as C  # noqa: E402

REGIONS = {"full": "full", "suffix": "suffix", "prefix": "prefix"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=C.TRAIN_SEEDS)
    parser.add_argument("--region", required=True, choices=list(REGIONS))
    args = parser.parse_args()
    from vhh_rl.online_pref.sigma_sampler import sample_region_sigma
    from vhh_rl.online_pref.trainer import OnlineArmConfig, run_online_arm

    cases = C.load_cases(C.TRAIN_CASES)
    design_positions = {cid: cases[cid].design_positions for cid in C.TRAIN_CASES}
    out_dir = C.RUN_ROOT / "phase1" / f"seed_{args.seed}" / f"T-{args.region}"

    def sample_sigma(pair, spec, cfg, student):
        return sample_region_sigma(student.structure_module,
                                   boundary_sigma=pair.branch_sigma,
                                   region=cfg.sigma_region, seed=spec.sigma_seed,
                                   device=C.DEVICE)

    cfg = OnlineArmConfig(
        arm=f"T-{args.region}", seed=args.seed, support="all",
        sigma_region=args.region, data_source="bank",
        bank_source_root=str(C.seed_dir(args.seed) / "B"),
        output_dir=str(out_dir), bank_root=str(out_dir))
    summary = run_online_arm(cfg, {"base_checkpoint": C.BASE_CKPT,
                                   "design_positions": design_positions,
                                   "sample_sigma": sample_sigma})
    print(json.dumps({"arm": cfg.arm, "seed": args.seed,
                      "reward_queries": summary["reward_queries_total"],
                      "updates": summary["updates"]}, indent=1))


if __name__ == "__main__":
    main()
