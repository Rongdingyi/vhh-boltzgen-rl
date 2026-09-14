# CF-OPSD Phase A Audit

任务书 §7-§9。全部为静态审计 + 实际 hook 验证（见 rollout parity）。

## 1. 基础信息

| 项 | 值 |
|---|---|
| BoltzGen commit | `a3149cf18eeb58648d1abbb27539bd73f746cdda` |
| base checkpoint | `/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt` |
| base checkpoint sha256 | `360af8bd6e59527ff6ec25dd81253967…` |
| sampling_steps | 50（native 面板一致） |
| schedule | 官方 `sample_schedule_af3`（`t_hats = sigma_tm*(1+gamma)`） |
| inference convention | 官方 CLI `--steps design`，EMA/raw 由 checkpoint 加载器决定（与 native 面板相同） |

## 2. Sampler 可用信号

| 信号 | 可用 | 说明 |
|---|---|---|
| `x0_coords_traj` | True | 每个 step 的 clean-output prediction（anchor candidate） |
| `coords_traj` | True | 每个 step 的 sampler state（非严格 network 输入） |
| `sample_atom_coords` | True | 最终输出 |
| 精确 network 输入 | **额外 hook `preconditioned_network_forward` 保存** | 任务书 §9：禁止猜；我们在 rollout 中保存 exact `noised_atom_coords` + sigma + `denoised` |

## 3. 硬解码与 mask

| 项 | 可用 |
|---|---|
| 官方 `res_from_atom14` | True |
| `fake_atom_mask` | True |
| `design_mask` | True |
| `atom_to_token` | True |
| `preconditioned_network_forward` | True |

## 4. 定义

```text
X_hat_0^(q) = anchors[q]（= 官方 x0_coords_traj[q]，由 network hook 精确保存）
X_q         = query_states[q]（= 官方 noised_atom_coords，进入 network 前）
sigma_q     = sigmas[q]（t_hat）
```

不重造 clean-output decoder；不使用 soft decoder；reward gradient 不在本阶段考虑。

## 5. 结论

- 所有需要信号在官方 sampler 中可获得且不需要修改采样方程。
- Phase A rollout 将保存 exact query state / anchor / endpoint 三类数据。
- **AUDIT PASS**
