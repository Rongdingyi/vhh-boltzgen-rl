#!/usr/bin/env python
"""Pre-Phase-1 sigma domain audit (task book §21)."""
from __future__ import annotations

import json
import statistics as st

import torch

import _common as C  # noqa: E402

N_MC = 20000
N_COND = 5000


def main() -> None:
    from vhh_rl.online_pref.sigma_sampler import (sample_region_sigma,
                                                  suffix_mass)
    from vhh_rl.online_pref.pair_bank import load_pair_bank, pair_bank_path
    from vhh_rl.native_atom14.checkpoint import load_base_model

    model = load_base_model(C.BASE_CKPT, device="cpu")
    sm = model.structure_module
    branch_sigmas = []
    for seed in C.TRAIN_SEEDS:
        bank = load_pair_bank(pair_bank_path(C.seed_dir(seed) / "B", seed, 1))
        branch_sigmas += [g["sigma"] for g in bank["groups"]]
    branch_sigma = st.median(branch_sigmas)
    q_b = suffix_mass(sm, branch_sigma)

    g = torch.Generator().manual_seed(C.TRAIN_SEEDS[0])
    z = torch.randn(N_MC, generator=g)
    sigma_mc = sm.sigma_data * (sm.P_mean + sm.P_std * z).exp()
    suffix_mc = float((sigma_mc <= branch_sigma).float().mean())

    def cond_stats(region: str) -> dict:
        values = []
        for i in range(N_COND):
            value = sample_region_sigma(sm, boundary_sigma=branch_sigma,
                                        region=region, seed=100000 + i, device="cpu")
            values.append(float(value.reshape(-1)[0]))
        values.sort()
        return {"median": st.median(values), "p10": values[int(0.10 * len(values))],
                "p90": values[int(0.90 * len(values))], "min": values[0],
                "max": values[-1], "n": len(values)}

    payload = {
        "protocol_sha256": C.protocol_hash(),
        "branch_progress": C.BRANCH_PROGRESS,
        "branch_sigma_median": branch_sigma,
        "branch_sigma_n": len(branch_sigmas),
        "train_sigma_median": float(torch.median(sigma_mc)),
        "suffix_mass_analytic": q_b,
        "suffix_mass_mc": suffix_mc,
        "prefix_mass_analytic": 1.0 - q_b,
        "conditional_suffix": cond_stats("suffix"),
        "conditional_prefix": cond_stats("prefix"),
    }
    payload["pass"] = (q_b >= 0.10 and (1.0 - q_b) >= 0.10)
    C.write_json(C.RUN_ROOT / "phase1" / "sigma_domain_audit.json", payload)
    print(json.dumps(payload, indent=1))
    if not payload["pass"]:
        raise SystemExit("sigma domain audit failed (mass < 0.10); STOP temporal")


if __name__ == "__main__":
    main()
