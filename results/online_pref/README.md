# online_pref results — archive

Task book: `ONLINE_PREFERENCE_SPATIOTEMPORAL_DPO_TASKBOOK.md`
Status: **Phase 0 PASS, Phase 1 FAIL → stopped per §26/§46/§47** (no Phase 2/3/4,
no 24-case / valid100 follow-up).

## Main tables

Phase 0 (3 seeds × {A offline diff_only, B online all-changed, V verified
control}, 8 held-out × 8 samples, eval seed offset 700000):

| seed | A | B | V | B−A | V−B | B≥A cases |
|---:|---:|---:|---:|---:|---:|---:|
| 20260915 | +2.438 | +2.877 | +3.307 | +0.439 | +0.430 | 6/8 |
| 43 | +2.907 | +3.253 | +2.091 | +0.347 | −1.162 | 2/8 |
| 44 | +2.913 | +3.371 | +2.888 | +0.458 | −0.483 | 5/8 |
| **mean ± stdev** | | | | **+0.415 ± 0.060** | −0.405 ± 0.799 | median 5 |

Phase 1 (same pair banks/schedules/ε/augmentation, only sigma domain differs):

| seed | T-full | T-suffix | T-prefix | suffix−full | suffix−prefix | prefix−full |
|---:|---:|---:|---:|---:|---:|---:|
| 20260915 | +3.632 | +2.954 | +3.103 | −0.678 | −0.149 | −0.528 |
| 43 | +4.360 | +3.902 | +3.974 | −0.458 | −0.072 | −0.386 |
| 44 | +2.529 | +2.764 | +3.635 | +0.235 | −0.871 | +1.105 |
| **mean ± stdev** | | | | **−0.300 ± 0.476** | **−0.364 ± 0.440** | +0.064 ± 0.905 |

Sigma domain audit: branch sigma 10.395; suffix mass 0.696 / prefix mass 0.304.
Validity: invalid 0.0% (Phase 0), 0.2% (Phase 1), FR mismatch 0 everywhere.
Query accounting: 128 scored sibling endpoints per 100-update arm
(32/round; the old "32" field counted pair records).

## Verdicts

- **Gate 0 PASS**: mean(B−A) = +0.415 ≥ +0.30, 3/3 seeds positive, median
  cases 5/8 ≥ 4, invalid 0, FR 0 → `artifacts/phase0_gate.json`.
- **Gate 1 FAIL**: mean(suffix−full) = −0.300 (< +0.25, 1/3 positive);
  mean(suffix−prefix) = −0.364 (< +0.50) → `artifacts/phase1_gate.json`.
- Verified/carrier filtering contributes nothing (V−B = −0.405, 1/3 positive).

## Component ordering that the data supports

1. Spatial support localization (train only changed design positions) — the
   stable core (see also `docs/paper_stage/P0_DIFFONLY_MULTISEED.md`).
2. Online sibling data refresh — small but stable (+0.415 ± 0.060), below the
   task book's +0.50 scaling threshold.
3. Everything else tested so far (signed local DPO, region/adaptive credit,
   sibling geometry distillation, geometry/carrier filtering, branch-aware
   sigma restriction) shows no cross-seed stable gain.

## Artifacts in this directory

- `phase0_seed_summary.csv`, `phase0_per_case.csv`, `phase1_seed_summary.csv`
- `artifacts/phase0_gate.json`, `artifacts/phase1_gate.json`
- `artifacts/phase0_eval.json`, `artifacts/phase1_eval.json`
- `artifacts/sigma_domain_audit.json`, `artifacts/p0_check.json`
- `artifacts/*_rounds.json` (per-arm round logs: queries, hamming, drift)
