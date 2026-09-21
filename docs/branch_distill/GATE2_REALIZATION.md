# Gate 2 — Teacher target realization

## 1. residue identity after transfer

- carrier all-match: 12/12 teacher endpoints reproduced downstream
## 2. query replay exactness

- max |pred - stored anchor| median 0.00e+00
## 3. local target error

- initial MSE median 3.091447
- final MSE median 0.026526
- mse_ratio median 0.009 (12/12 <= 0.5)
## 4. downstream sequence

- hamming improved 10/12 (median 14.5 -> 9.0)
## 5. reward safety

- median(postfit - peer) = +3.362, safe 10/12
## 6. 失败发生在哪一级

- A_replay: PASS
- B_target: FAIL
- C_learnability: PASS
- D_downstream: PASS
- E_safety: PASS

verdict: **FAIL**

protocol_sha256: `c45308687969f675664684c48ae5ec0b261c1d47dc5a47bd2cb56c37a9bbdb29`
