# CF-OPSD Related Work：边界与差异化（task book §1）

## 1. OPSD for LLMs

```text
Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models
```

核心：student on-policy trajectory + teacher/enriched context + distribution
matching + self-distillation。仅作为 OPSD 概念来源，不是 diffusion 实现。

## 2. DiffusionOPSD（2026）

```text
On-Policy Self-Distillation in Diffusion Models
```

机制：frozen behavior policy → on-policy denoising trajectory → low-noise
query state → clean-output anchor → differentiable reward gradient →
bounded positive/negative target → detached target fitting → finite update
budget → EMA behavior refresh → repeat。

关键依赖：`∇_y R(y)`，即 reward 对 clean-output **可微**。

## 3. 我们与 DiffusionOPSD 的核心差异

BoltzGen native design 的 reward 路径：

```text
X_atom14 --res_from_atom14(硬几何解码)--> S --R(黑盒 scorer)--> reward
```

`res_from_atom14` 是硬解码（fake-atom geometry → nearest backbone atom →
placement count → lookup table → AA identity），因此 `∇_X R(g(X))` 不可得。

**本任务书禁止第一版把 decoder 换成 soft/differentiable 版本。**

差异化定位（§97）：

> Existing diffusion OPSD methods construct targets from differentiable
> reward gradients. Protein all-atom co-design is a different regime: the
> reward is defined on discrete sequences recovered by a non-differentiable
> geometric decoder. We construct local clean-output targets from **black-box
> counterfactual outcome interventions** rather than reward gradients.

## 4. 与 CF-DPO 的关系（§98）

| | 更新机制 |
|---|---|
| CF-DPO | counterfactual credit → weighted preference likelihood (DPO) |
| CF-OPSD | counterfactual credit → explicit bounded geometric target → self-distillation |

同一 credit estimator、两种 policy-update 范式；只有 CF-OPSD 显著更好才升为主方法，
否则 CF-DPO 保持主线。
