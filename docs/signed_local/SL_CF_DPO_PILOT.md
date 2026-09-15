# SL-CF-DPO Pilot（task book §88）

## 1. Edge dataset

```json
{
 "split": "train",
 "n_pairs_used": 192,
 "n_sites_considered": 1480,
 "n_both_negative": 484,
 "n_sign_flip": 996,
 "n_drop_attempted": 1480,
 "n_drop_accepted": 1191,
 "n_gain_attempted": 1480,
 "n_gain_accepted": 1214,
 "acceptance_by_context": {
  "winner_drop": 0.8047297297297298,
  "loser_gain": 0.8202702702702702
 },
 "preferred_anchor_fraction": 0.5014553014553015,
 "preferred_cf_fraction": 0.4985446985446986,
 "median_abs_dR": 0.8218234777450562,
 "median_moved_rms": 2.7133376598358154,
 "fr_mismatch": 0,
 "invalid": 0,
 "n_edges": 2405,
 "acceptance_by_class": {
  "both_negative": 0.8966942148760331,
  "sign_flip": 0.8945783132530121
 }
}
```

Held-out edges：

```json
{
 "split": "heldout",
 "n_pairs_used": 4,
 "n_sites_considered": 30,
 "n_both_negative": 10,
 "n_sign_flip": 20,
 "n_drop_attempted": 30,
 "n_drop_accepted": 19,
 "n_gain_attempted": 30,
 "n_gain_accepted": 18,
 "acceptance_by_context": {
  "winner_drop": 0.6333333333333333,
  "loser_gain": 0.6
 },
 "preferred_anchor_fraction": 0.43243243243243246,
 "preferred_cf_fraction": 0.5675675675675675,
 "median_abs_dR": 0.7580280303955078,
 "median_moved_rms": 2.519641637802124,
 "fr_mismatch": 0,
 "invalid": 0,
 "n_edges": 37,
 "acceptance_by_class": {
  "both_negative": 0.7,
  "sign_flip": 0.7
 }
}
```

Edge validation：`not run`

## 2. Manifold audit

```json
{
 "n_edges": 2405,
 "n_audited": 2405,
 "p99_outlier_fraction": 0.015384615384615385,
 "gate": "PASS",
 "native_mean_by_case": {
  "sab2_1acy_h": 0.5839941999898125,
  "sab2_2f58_h": 0.752536394201968,
  "sab2_4i1n_b": 0.5821244920039703,
  "sab2_4r2g_d": 0.9681815639083879,
  "sab2_4xmp_h": 1.156061370569167,
  "sab2_5vxk_b": 0.5638152571020404,
  "sab2_6mqe_h": 0.6463997527490525,
  "sab2_6p4b_b": 0.4500077750060096,
  "sab2_6u52_c": 0.8078759285592696,
  "sab2_7jkt_h": 0.5822723131990605,
  "sab2_7nqk_b": 0.6829673059905872,
  "sab2_7qxd_kkk": 0.5313464708486938,
  "sab2_7sl5_d": 0.8477335131595396,
  "sab2_7x7d_b": 0.612776678399238,
  "sab2_8bcz_h": 0.457729538985537,
  "sab2_8c3k_b": 0.37046553119241804,
  "sab2_8en1_d": 0.6306342707147413,
  "sab2_8en5_e": 0.6805321630070011,
  "sab2_8gb6_h": 0.5078803560636516,
  "sab2_8k33_b": 0.5992711664645564,
  "sab2_8pnu_d": 0.6976923230867745,
  "sab2_8rz0_f1": 0.9677365632045584,
  "sab2_9b6s_h": 0.5188576743938901,
  "sab2_9ubr_g": 0.4380808366501583
 }
}
```

## 3. One-edge overfit

```json
{
 "n_edges": 12,
 "n_positive_final": 12,
 "median_dz": 0.3456175923347473,
 "nonfinite": 0,
 "gate_pass": true
}
```

## 4. Local SNR audits

gradient-direction：`{"n_edges": 32, "positive_fraction": 0.96875, "median_dz": 0.0009753182530403137, "gate": true}`

sigma robustness：`{"n_edges": 32, "median_positive_fraction": 0.9375, "gate": true}`

## 5. Pilot main table（held-out 4 cases x 8 samples, 固定 eval seed）

| arm | global steps | local steps | reward | Δbase | ΔCF | local acc | invalid | FR | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0 | 0 | -1.206 | +0.000 | -4.861 | 0.500 | 0.00% | 0 | 32 |
| CF-mini (A1) | 100 | 0 | +3.654 | +4.861 | +0.000 | 0.432 | 0.00% | 0 | 32 |
| Local-only (A2) | 0 | 100 | +0.362 | +1.568 | -3.292 | 0.441 | 0.00% | 0 | 32 |
| CF+Neg | 75 | 25 | +3.618 | +4.824 | -0.036 | 0.550 | 0.00% | 0 | 32 |
| **SL-CF-DPO (A4)** | 75 | 25 | +2.446 | +3.653 | -1.208 | 0.532 | 0.00% | 0 | 32 |
| DirectionShuffle (A5) | 75 | 25 | +1.850 | +3.056 | -1.804 | 0.541 | 0.00% | 0 | 32 |
| CF+Local add-on (A6) | 100 | 25 | +3.420 | +4.627 | -0.234 | - | 0.00% | 0 | 32 |
| CF-125 compute-matched (A7) | 125 | 0 | +4.502 | +5.709 | +0.848 | - | 0.00% | 0 | 32 |

## 6. Held-out local preference accuracy（3 seeds）

| arm | acc (mean ± sd) | median z | both-neg | sign-flip | winner-drop | loser-gain |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0.500 ± 0.000 | +0.00000 | 0.500 | 0.500 | 0.500 | 0.500 |
| CF-mini (A1) | 0.432 ± 0.058 | -0.00314 | 0.364 | 0.346 | 0.263 | 0.444 |
| Local-only (A2) | 0.441 ± 0.025 | -0.00140 | 0.455 | 0.385 | 0.316 | 0.500 |
| CF+Neg | 0.550 ± 0.067 | +0.00178 | 0.545 | 0.423 | 0.368 | 0.556 |
| **SL-CF-DPO (A4)** | 0.532 ± 0.092 | +0.00147 | 0.545 | 0.346 | 0.316 | 0.500 |
| DirectionShuffle (A5) | 0.541 ± 0.076 | +0.00087 | 0.455 | 0.500 | 0.421 | 0.556 |


## 7. Frozen protocol

```yaml
version: 1
method: sl_cf_dpo
repo_commit: a0484a718646a5fe57cae71eff7525b0fbb5b07e
boltzgen_commit: a3149cf18eeb58648d1abbb27539bd73f746cdda
base_checkpoint_sha256: 360af8bd6e59527ff6ec25dd81253967f3bd3567d200053b10680634751f8e3c
pairs_train_sha256: fd9776fd8e82d2be57904b5f82ddd84cbfae4831d27e8ac54eb2713aae338b39
residue_credit_sha256: c8a1b01abfbaf45e7040d36dea9927973a31f0be1963eb45d1fad73fbb827305
counterfactual_scores_sha256: e805a59cc96a958e6a003172cde965272fa7b6d8c576b5337446a2a7f73b3f9d
residue_weights_sha256: b5d1cbe95e75e99593850627edfe82f2f69f5d531242d5ca03b16192e89448d3
train_edges_sha256: 0373b1b151c5d548cdebc6dacd72e5e7bc5e7b2a5c42784fa1b129e6bebf521f
heldout_edges_sha256: null
train_cases:
- sab2_6u52_c
- sab2_7sl5_d
- sab2_7nqk_b
- sab2_6mqe_h
heldout_cases:
- sab2_4hf5_h
- sab2_4mwf_h
- sab2_5mp6_h
- sab2_6cvk_b2
reward_tolerance: 0.05
beta_global: 10.0
beta_local: 10.0
eta: 0.75
lr: 1.0e-05
schedule: deterministic_3_to_1
eval_seeds: seed_base + 800000 (8 samples/case)

```

## 8. Decision

见 `SL_CF_DPO_DECISION.md`（GO/NO-GO 条件见 task book §48/§49）。
