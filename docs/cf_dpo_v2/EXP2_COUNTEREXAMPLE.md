# CF-DPO v2 — Experiment 2: two-residue counterexample (proposal §3.1)

Reward table: {'00': 0, '01': 2, '10': 2, '11': 1}

| site | c_drop | c_gain | c_avg | c_cons |
|---|---|---|---|---|
| 1 | -1.0 | +2.0 | +0.5 | 0.0 |
| 2 | -1.0 | +2.0 | +0.5 | 0.0 |

- endpoint gap F(11)-F(00) = +1.0
- averaged decomposition residual = +0.0e+00 (exact)
- c_cons is zero at every site -> uniform-floor fallback
- signed local edges keep: 11->01 ΔR=+1, 11->10 ΔR=+1 (both improve),
  so a signed comparison model learns that the global winner is not
  locally optimal, which no nonnegative weighting can express.
