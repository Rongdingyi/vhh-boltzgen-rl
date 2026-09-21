# Gate 1 — Same-state sibling branching

## 1. K=8 是否产生不同序列？

- progress 0.60: unique median 8.0, teacher-vs-median hamming median 14.5
- progress 0.70: unique median 8.0, teacher-vs-median hamming median 13.5
- progress 0.80: unique median 8.0, teacher-vs-median hamming median 6.5
- progress 0.90: unique median 1.0, teacher-vs-median hamming median 0.0

## 2. reward spread

- progress 0.60: reward_std median 4.195, best_minus_median median 4.707
- progress 0.70: reward_std median 3.063, best_minus_median median 4.968
- progress 0.80: reward_std median 2.006, best_minus_median median 3.106
- progress 0.90: reward_std median 0.000, best_minus_median median 0.000

## 3. 最早可用 progress

selected_progress = 0.6

## 4. teacher vs median 平均差几个设计位点

14.5 (median across groups)

## 5. validity

- progress 0.60: valid_rate median 1.000
- progress 0.70: valid_rate median 1.000
- progress 0.80: valid_rate median 1.000
- progress 0.90: valid_rate median 1.000

## 6. FAIL 的直接原因

- Gate 1 PASS

protocol_sha256: `c45308687969f675664684c48ae5ec0b261c1d47dc5a47bd2cb56c37a9bbdb29`
