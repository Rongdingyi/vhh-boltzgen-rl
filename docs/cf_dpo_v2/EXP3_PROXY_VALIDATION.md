# CF-DPO v2 proxy validation（修正后最终版）

在 v2-u100 checkpoint（warmup-restart 校准，κ_eff=9.34e3）上，对比较图的
非 same-seq 边做**分层随机抽样**（seed 固定），ground truth 由节点 reward
重算 `R(b)-R(a)`，nσ=32。

| 指标 | 值 |
|---|---|
| 评估边数（dR≠0） | 78 |
| stored-vs-node label mismatch | **0** |
| **代理符号一致率 vs node reward** | **0.615** |
| median \|Δh\|（信号） | 4.01e-05 |
| median σ-噪声（Δh 的 std） | 2.70e-04 |

结论：修正所有图/标签 bug 后，denoising-energy proxy 的局部排序仍**不可靠**
（符号一致率 0.615，噪声约为信号的 6.7 倍）。这解释了为什么基于该代理的
signed/v2 BCE 训练无法兑现（机制性失败），也与 proposal §11.2 标注的
"denoising loss 不是精确 log density" 风险一致。
