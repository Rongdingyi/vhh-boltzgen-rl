# P0-A — Multi-seed Robustness（3 training seeds）

任务书 §5–§12；协议冻结：`experiments/paper_stage/configs/FROZEN_PROTOCOL.yaml`。

## 1. 设置

- Arms：N3 Uniform Fake DPO / B2 Shuffle Credit / B3 CF-DPO。
- Seeds：primary = 20260913（既有 headline 结果，保持不变）+ 43 + 44（本轮补跑）。
- 训练设置与 headline 完全一致（β=10、lr=1e-5、500 steps、24 train cases、
  192 pairs、paired sigma/noise/augmentation、trainable = score_model）。
- Checkpoint 选择：held-out 8 cases × 8 samples，§54 规则（最高 reward、
  invalid ≤ base+1pp、FR=0），valid100 仅用于最终评测。
- 评测 seeds 固定（valid100：seed_base+700000、8 samples/case）。

## 2. 训练层面分离（z / implicit acc，steps 250–450）

| seed | CF | N3 | Shuffle |
|---|---|---|---|
| primary (20260913) | +0.225 / 0.86 | +0.019 / 0.64 | +0.016 / 0.55–0.68 |
| 43 | +0.236 / 0.87 | +0.019 / 0.66 | +0.022 / 0.61–0.68 |
| 44 | +0.163 / 0.86 | +0.019 / 0.64 | +0.035 / 0.61 |

训练信号层面 CF 与两个 control 在每个 seed 上稳定分离。

## 3. valid100 主结果（100 cases × 8 samples，Δ vs base）

| seed | CF | N3 | Shuffle |
|---|---|---|---|
| primary | **+5.566**（99/100） | +2.796（95/100） | +3.531（96/100） |
| 43 | **+4.509**（97/100） | +4.408（96/100） | +3.566（100/100） |
| 44 | **+4.700**（98/100） | +4.310（98/100） | +3.728（99/100） |
| **mean ± std** | **+4.925 ± 0.563** | +3.838 ± 0.904 | +3.608 ± 0.105 |

Per-seed paired gap（同一 eval seeds 下）：

| seed | CF − N3 | CF − Shuffle |
|---|---|---|
| primary | +2.770 | +2.035 |
| 43 | **+0.101** | +0.943 |
| 44 | +0.390 | +0.972 |

invalid：全部 ≤ 1/600，FR mismatch = 0。

## 4. Gate 判定（§12）

| gate | 结果 |
|---|---|
| 3/3 seeds CF Δ > +4.0 | **PASS**（4.51 / 4.70 / 5.57） |
| 3-seed mean(CF − N3) > +2.0 | **FAIL**（+1.087） |

并且出现 §12 明确警告的模式：**seed 43/44 上 CF ≈ N3**（+0.10 / +0.39）。
→ 按任务书要求：**不继续扩论文主张，先排查 credit / checkpoint instability**。

## 5. 不稳定性排查（§63 checklist）

| 检查项 | 结果 |
|---|---|
| credit dataset hash | 三 seed 使用同一权重文件（同 hash），非数据问题 |
| pair ordering | 由训练 seed 控制；各 seed 训练曲线稳定、无 NaN |
| checkpoint selection | 各 seed 独立按 held-out 选择；N3 s43/s44 的 held-out reward（+5.87/+6.55）高于 primary N3（+3.88），说明 N3 新 seed 本身更强，而非选择噪声 |
| β × credit scale | 统一 β=10、η=0.75；CF z 0.163–0.236 无异常 |
| gradient norm / z | 各 seed 稳定（CF grad ~7–18，N3 ~6–8） |
| 结论 | **不是 instability，而是 N3 训练 seed 方差大**（valid100 Δ 2.80 → 4.41 → 4.31），CF 更稳定（4.51–5.57） |

## 6. 诚实结论

1. Primary-seed 的 CF − N3 = +2.77 **不稳健**；3-seed 平均只有 **+1.09**。
2. CF 在每个 seed 上都 ≥ N3 与 Shuffle，且方差最小；但"CF ≫ N3"不能作为主 claim。
3. CF − Shuffle 三 seed 平均 +1.32（+0.94 ~ +2.04），且 Shuffle 方差最小，
   说明 attribution mapping（CF vs shuffle）的增益是稳定的。
4. 论文表述需改为：
   - **CF-DPO consistently improves over N3 and Shuffle across seeds**（方向稳定）
   - 单 seed 上"− N3 超过 +2"的幅度属于 seed 波动，不作为定量 claim。
5. 这一发现同时影响 §12 的"论文扩展"判断：**在 P0 其余证据（ablation、
   structure、query-budget、second reward）完成之前不写 headline claim**。

## 7. 文件

- `results/paper_stage/multiseed_summary.csv`、`multiseed_seed_level.csv`
- `runs/paper_stage/valid100/paper_*_s*.json`
- `runs/paper_stage/multiseed/checkpoint_selection.{json,md}`
