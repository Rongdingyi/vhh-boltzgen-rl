# 从反事实权重到结果空间对齐：CF-DPO 的算法升级方案

## 结论

建议暂停 CF-OPSD，将现有 CF-DPO 保留为不可覆盖的基线。下一版不以自适应 η、Shapley 权重或增加结构奖励为核心，而改为 **Counterfactual Preference Correction with Outcome-Consistent Alignment**：保留反事实查询中的背景与正负方向，直接学习局部偏好；同时把一个序列对应的多种合法原子构象作为同一个奖励结果处理，避免模型把某一套偶然生成的坐标误当作奖励目标。

本文简称“CF-DPO v2”，不预先包装新缩写。建议论文题目为：

**Beyond Credit Weights: Counterfactual Preference Correction for Protein Co-Design**

这是一个尚未进行原生蛋白训练验证的研究方案，不是已证明优于 CF-DPO 的结果，更不是录用保证。其优势在于：问题来自现有原始数据，优化目标可推导，区别于单纯加权的实验可以明确设计，失败条件也清楚。

## 1. 现有结果支持什么

`cfdpo_paper_stage_report.zip` 的 `docs/PAPER_STAGE_RESULTS.md` 记录：CF-DPO 三个训练种子的 reward 增量为 +5.566、+4.509、+4.700；N3 为 +2.796、+4.408、+4.310。均值差为 +1.087，而不是 primary seed 的 +2.770。Diff-only、Drop-only、Gain-only、CF 的 primary-seed 增量分别为 +4.221、+4.932、+5.258、+5.566。[D1]

这支持“反事实信息值得利用”，但还不能证明双向机制对单向机制稳定占优。后续最重要的基线是 **Gain-only 与 Diff-only**，不能只与较弱的一次 N3 结果比较。

结构方面，100-case Boltz-2 的 CDR RMSD 为 N3 1.780 Å、CF 1.854 Å；差值 +0.074 Å 的区间为 [−0.056,+0.202]。不能把未显著恶化解释为已经证明非劣，也不能继续讲“CF 改善结构”。Protenix-v2 的 N3 排序同样优于 CF。[D1]

以下新增统计来自对已有 CSV 的重新计算，不是新增模型实验，也不是对仓库实现的完整代码审计。统计输入、哈希、脚本和结果随本文件提供。[D2–D4]

## 2. 比信用稀疏更关键的新发现

### 2.1 许多局部偏好与全局 winner 方向不一致

在 24 个训练 case、192 对偏好样本、2,996 个不同残基事件中：

| 类型 | 判定 | 数量 | 占全部事件 |
|---|---|---:|---:|
| 两个背景都支持 winner 残基 | drop>0 且 gain>0 | 1,375 | 45.89% |
| 两个背景都支持 loser 残基 | drop<0 且 gain<0 | 540 | 18.02% |
| 偏好方向随序列背景翻转 | drop 与 gain 异号 | 1,081 | 36.08% |

使用 0.05 分作为纯诊断容差、要求两个差值绝对值均超过该容差后，后两类仍分别有 484 个和 996 个，占全部事件的 16.15% 和 33.24%。因此现象不只是接近零的数值抖动。[D2–D4]

这些是**固定 classifier 的局部函数响应**，不能称为生物实验中的因果贡献。存在背景翻转，说明一个与背景无关的非负重要性权重，不足以表达该残基在两个背景下分别应当如何更新。

### 2.2 现有查询已经找到了比 winner 更高分的序列

`c_drop<0` 表示把 winner 某一位换回 loser 的残基，反而提高 reward。以提升超过 0.05 分计：

- 共有 1,020 条此类反事实查询；
- 192 对样本中有 190 对至少包含一条；
- 每对样本最佳单点回退的提升中位数为 **1.3486 分**，均值为 1.6354 分。

换言之，很多 winner 只是原始采样池中的 winner，不是查询邻域中的最优序列。现有算法已经支付查询成本，得到更好的候选，却主要将这些数据压缩成原 winner/loser 的权重。[D2–D4]

这一发现支持把研究问题从“哪些位置重要”改为：

\[
\text{全局优选标签，应该如何被修正为具有背景依赖性的局部偏好？}
\]

### 2.3 对旧稀疏性结论的修正

“top 30% 残基占 94% credit”中的 credit 是经过双方为正筛选和 `min` 截断的 `c_cons`。重新以每对 `ceil(0.3*d)` 个位置计算，top30 质量的中位数为：

| 归因量 | top30 质量中位数 |
|---|---:|
| 正值保守量 c_cons | 94.01% |
| abs(c_avg) | 66.48% |
| abs(c_drop) | 66.64% |
| abs(c_gain) | 65.97% |

因此 94% 不是“解释了真实 reward gap 的 94%”，也不能独立证明真实奖励函数只有极少数重要残基。信用不均匀确实存在，但之前的强稀疏叙述含有估计器自身截断产生的集中效应。[D2–D4]

## 3. 为什么之前的理论升级不够

之前提出的双向均值

\[
\phi_i=\tfrac12(c_i^{drop}+c_i^{gain})
\]

在二值化 winner/loser 残基组合、且只有一阶与二阶交互的函数上，确实可以分解端点差值。但要注意三点。

第一，该性质属于均值估计器，不属于当前实际使用的 `min + 正值截断 + uniform floor`。

第二，双向/配对采样在低阶交互下的精确性已有 Shapley 文献，不能把这一基本性质当作独有理论贡献。[R6]

第三，**精确解释端点差值，不等于给出了正确的局部改动方向**。下面的例子直接说明这一点。

### 3.1 一个两残基的反例

令 loser 为 00、winner 为 11，0/1 分别表示选择 loser/winner 的残基，奖励为：

\[
F(z_1,z_2)=2z_1+2z_2-3z_1z_2.
\]

| 序列 | reward |
|---|---:|
| 00 | 0 |
| 10 | 2 |
| 01 | 2 |
| 11 | 1 |

全局确实有 11>00。但两个位置均有：

\[
c_i^{gain}=2,\quad c_i^{drop}=-1,\quad \phi_i=0.5.
\]

于是 `sum(phi)=1=F(11)-F(00)`，分解残差严格为零；然而在 11 的背景下，回退任意一位都更好。实际保守量 `c_cons` 在两个位置都为零，只剩 uniform fallback，恰好丢掉最有价值的信息。

这个反例只有二阶交互，不需要复杂高阶上位性。用端点分解残差决定自适应 η，无法解决它。

此外，`10 z1 + z1 z2 z3 - z1 z2 z4` 含有非零三阶项，但端点分解残差仍可相互抵消。因此残差只能叫端点重构误差，不能直接叫“高阶交互强度”或“归因可信度”。上述反例及后文恒等式已经由随附脚本在精确有限状态模型中检查。[D4]

## 4. 文献边界

| 近邻工作 | 已有内容 | 新方案必须避免的重复表述 |
|---|---|---|
| ResiDPO / EnhancedMPNN | 利用局部结构反馈做残基级偏好与约束学习 | 不能说首次蛋白残基级 DPO |
| DecompDPO | 在分子扩散中利用可分解性质进行子结构偏好优化 | 不能说首次局部化 diffusion-DPO |
| Beyond Uniform Credit | 用反事实遮蔽获得局部 credit 并分配策略梯度 | 不能把反事实重要性加权本身当作新范式 |
| TokenRatio / TBPO | token-level conditional density-ratio matching | 不能把局部密度比匹配单独当作新发明 |
| Di3PO | 固定无关背景构造更局部的图像扩散偏好对 | 不能只靠“背景相同的偏好对”建立新颖性 |
| Shapley 配对采样 | 低阶交互下精确性与加性恢复 | 不能包装基础归因性质为新定理 |

本方案应守住的边界是：**任意黑盒序列奖励 + 局部偏好方向翻转 + 连续几何经多对一解码产生离散结果**。因此既要纠正偏好，又要处理同一个奖励结果有多个几何实现的问题。[R1–R6]

## 5. 主算法原则：对齐序列结果，而不是偶然的几何代表

令 c 为设计条件，X 为原生全原子设计状态，S=g(X) 为官方硬解码得到的序列。固定参考分布 p0(X|c)，奖励始终为 R(S)。在同一合法支持集和固定刚体坐标规范下考虑：

\[
J(q)=\mathbb E_q[R(g(X))]-\beta KL(q(X|c)\Vert p_0(X|c)).
\]

这是标准 KL 正则化奖励优化形式，不是本方案发明；其 Gibbs 形式也是已有 DPO 理论的基础。[R7]

最优解为：

\[
q^*(X|c)=\frac{p_0(X|c)e^{R(g(X))/\beta}}{Z(c)}.
\]

由于奖励只依赖序列，直接得到：

\[
\boxed{q^*(X|S,c)=p_0(X|S,c).}
\]

其含义不是“不允许结构变化”，而是：**当目标只要求更偏好某些序列时，应重分配序列的概率；对同一个序列内部各种构象，没有额外奖励依据去扭曲其参考分布。**

### 5.1 对优化损失的有用分解

\[
\begin{aligned}
J(q^*)-J(q)
&=\beta KL(q(X|c)\Vert q^*(X|c))\\
&=\beta KL(q(S|c)\Vert q^*(S|c))\\
&\quad+\beta\mathbb E_{S\sim q}KL(q(X|S,c)\Vert p_0(X|S,c)).
\end{aligned}
\]

这把误差分为“序列偏好没有学对”和“同一序列内部多做了几何分布改动”。不是新造一个结构奖励，也不是把 refold RMSD 混进总分。

### 5.2 为什么这不保证平均 refold 指标不变

即使条件结构分布被完美保留，新的序列边缘分布也可能偏向更难折叠的序列。因此总体平均 refold RMSD 仍可变差。该原则只能控制**奖励没有要求的额外条件几何扭曲**，不能修复 classifier 与真实结构/功能目标的不一致。

后续论文必须坚持这一区别。

## 6. 模块一：把反事实查询变成有方向的局部比较

不再将 `drop` 与 `gain` 压缩成单一非负权重。

对于每个原始 pair 和位点 i，直接保留：

\[
(S^-,S_i^{-\leftarrow+},c_i^{gain}),
\]

\[
(S_i^{+\leftarrow-},S^+,c_i^{drop}).
\]

每个三元组前两个序列只改一个位点，第三项是“后者减前者”的实际 reward 差。若为负，就学习反方向；若前后两个背景方向相反，两条边都保留。

把已有更高分的反事实序列作为真正的比较节点，不再仅把它们用于原始 pair 的权重。依旧保留原来的全局偏好边，并利用同 case 已打过分的原始样本建立连接，避免每个局部子图只能确定自己的独立常数。

这里“图”只表示一个比较数据表，不训练 GNN。每个节点是完整序列和设计条件，每条边有精确的 classifier 差值。不能将不同 case 或不同条件的边混起来。

无需假设奖励加性或只有二阶交互，因为每条边都在完整且固定的序列背景下查询。局限是：它只监督访问过的背景，不会神奇推断未查询的全部交互组合。

## 7. 模块二：用多个几何实现表示一个序列

### 7.1 必须先有合法的几何实现

当前反事实 CSV 只有序列与分数，不自动拥有该序列对应的合法 atom14 结构。不能只替换残基标签、保留与旧氨基酸不匹配的坐标，也不能把 winner 的虚拟原子直接粘贴到 loser 上。

BoltzGen 的 atom14 中包含真实骨架、真实侧链以及编码残基类型的虚拟原子。不能把后十个槽位一概等同为同一种可自由替换的 fake geometry；虚拟原子配置也随氨基酸改变。[R9]

首选审计本地模型已有的固定序列条件/局部补全能力；若不能直接使用，则只做小规模参考模型局部修复与硬解码拒绝采样。修复路径可以参考扩散 inpainting，但 RePaint 的一般思路不等于已有 BoltzGen 正确实现。[R10]

每条拟训练的反事实边，需要合法几何 X_a、X_b 满足：

\[
g(X_a)=S_a,\quad g(X_b)=S_b.
\]

它们的 FR 和非编辑背景尽量匹配，所有编辑位置、物理侧链变化和虚拟槽位变化都明确记录。失败与拒绝不能静默丢掉，必须按残基类型和偏好正负类别报告。

**生成数据时使用目标序列约束，与训练时的条件 c 是两回事。训练两端必须使用原任务相同的条件，不能把答案序列泄漏给其中一端。**

### 7.2 同序列的多个几何实现

对选定序列 S 收集至少两个合法几何实现：

\[
X_1^S, X_2^S,\quad g(X_1^S)=g(X_2^S)=S.
\]

它们形成“同结果边”：reward 差严格为零。首先使用参考模型局部修复产生的合理邻域构象；不能仅靠旋转和平移凑数。

**主实现不假设这些构象是 p0(X|S,c) 的精确条件样本。** 在同一序列的任意合法几何点上，理想最优密度比都应该相等；因此可用点对约束检验必要条件，而不必一开始求难以计算的序列边缘似然。覆盖不足依然是局限，必须明确。

同一个模型“输入指定序列再预测结构”的模式，也不自动等于原生生成分布在 g(X)=S 条件下的采样。两种模型调用改变了条件分布，不能混同。

不同构象复用同一条 sequence reward，不增加 classifier 查询，但增加参考扩散计算。必须记录该成本。

## 8. 主实现：统一的有符号比较与同结果约束

定义 reference-relative potential：

\[
H_\theta(X,c)=\beta\log\frac{p_\theta(X|c)}{p_0(X|c)}.
\]

数据图的节点是合法原子设计 X，每个节点有确定的解码序列 S。比较边有三类：

1. 原始全局边，用于保持不同候选子图之间的连通；
2. 固定序列背景的单点反事实边，实际差值可以正或负；
3. 同序列不同构象边，实际差值为零。

它们统一为 e=(X_a,X_b,c)：

\[
\Delta R_e=R(g(X_b))-R(g(X_a)).
\]

令：

\[
t_e=\operatorname{sigmoid}(\Delta R_e/\tau),
\]

\[
z_e=[H_\theta(X_b,c)-H_\theta(X_a,c)]/\tau.
\]

主训练目标只有一个：

\[
\boxed{\mathcal L=\mathbb E_{e\sim\mathcal E}\operatorname{BCEWithLogits}(z_e,t_e).}
\]

同结果边的 t=0.5，其最优条件是两个构象的相对密度变化一致。不是强迫它们的原始概率相同，也不是把它们的坐标拉到一起，更不是将所有原子冻结。

不同边类型的采样比例要在开发协议中冻结。按条件和序列对分层采样，避免一个序列构象多就天然获得更大权重。可以以局部边为主、保留全局连接和同结果边，再做去掉同结果边的消融；不要把采样比例伪装成无超参。

不再需要 c_cons、η=0.75、负 credit 截断或 uniform fallback。弱差值形成接近0.5的软标签。t_e是根据固定reward差构造的训练目标，不声称它是实测的人类偏好概率。若奖励绝对标度不可靠，应先做基于训练集固定尺度的校准；不能每条边单独 z-score。

不要用负 credit 乘 MSE，那可能产生无界回归目标。这里负号仅改变偏好方向。sigmoid、软标签和密度比本身不是新发明；区别在于带背景的反事实纠错，以及硬解码同结果的相对概率约束。

## 9. 边缘化解释：为什么同结果约束不是另一个普通anchor

主实现不需要显式计算下面的积分。它用于说明为何这个约束有明确的概率目标。

记 aθ=Hθ/β。在精确参考条件分布下：

\[
\frac{p_\theta(S|c)}{p_0(S|c)}
=\mathbb E_{X\sim p_0(X|S,c)}e^{a_\theta(X,c)}.
\]

令 vθ(S) 为右侧期望的对数，则：

\[
v_\theta(S)-\mathbb E_{p_0(X|S,c)}a_\theta(X,c)
=KL(p_0(X|S,c)\Vert p_\theta(X|S,c)).
\]

所以同序列不同几何的相对密度比不一致，正是条件几何分布额外偏移的一种表现。点对等价约束是对此必要条件的稀疏可计算实现，而不是声称已计算出精确条件 KL。

若未来获得可靠的参考条件采样，才值得做显式边缘化版本：

\[
\widehat v_\theta(S)=\operatorname{logsumexp}_{m=1}^M a_\theta(X_m^S)-\log M.
\]

但这一版本**不作为第一轮必做组件**。有限M对数估计有偏；近似条件采样增加另一层误差；M=1也不能测条件几何差异。先用同结果边验证原则，比同时引入难以验证的边缘采样假设更稳妥。

## 10. 可以证明的性质及边界

### 10.1 条件结构保持

精确 Gibbs 目标对每个序列保持 p0(X|S,c)。证明只需将 p0(X|c)e^{R(S)/β}/Z 按固定 S 条件化，序列权重被约掉。

### 10.2 任意交互下的局部一致性

每条反事实边直接使用完整背景下的 R(S_b)−R(S_a)。因此不需要把一个全局差值加性拆到残基。上位性交互可以任意，但结论只限于已查询背景和合法支持。

### 10.3 理想有限支持下的目标识别

若每个条件的有限几何节点比较图连通，参考概率严格为正，密度比可精确计算，且模型族可实现目标，则软比较项要求：

\[
H_\theta(X_b)-H_\theta(X_a)=R(g(X_b))-R(g(X_a)).
\]

连通性给出 Hθ(X)=R(g(X))+常数；归一化决定该常数。在这个有限支持模型中得到 pθ(X|c)=q*(X|c)。同结果边直接要求同序列的相对密度变化一致。若要推广到完整连续分布，需要相应的支持覆盖或稠密一致性假设。

这不是对实际非凸训练的全局收敛保证，不是对未见序列的保证。图连通、覆盖和精确似然是明确的理论假设；不能把有限图识别结论推广成原生模型的全局分布保证。

### 10.4 不应增加没有信息量的“环路损失”

若所有边使用同一个节点分数 Hθ(X)，沿闭合路径的分数差本来就会相消。再加 cycle-consistency loss 只是恒等式，不是新学习信号。关键是节点分数来自同一生成模型、边标签来自真实反事实查询，而不是人为给每条边独立制造分数。

## 11. 如何接到 BoltzGen，而不偷换为 OPSD

### 11.1 复用的部分

保留原生 atom14 sampler、官方硬解码、固定 scorer、数据划分、checkpoint加载、reference与policy、反事实缓存和重折叠评测。继续只训练 `structure_module.score_model`。

新增的是比较数据构造、合法反事实几何补全、同序列多构象比较和新 loss。不需要对 scorer 求梯度，不训练新 reward model，也不需要 EMA behavior 或每轮 on-policy rollout。

### 11.2 最大数学风险：denoising loss 不是精确 log density

Diffusion-DPO 使用的是由扩散训练/变分推导获得的可计算代理，不是任意一个 masked MSE 就能当作精确 log p。[R8]

工程版可先定义：

\[
\widehat a_\theta(X,c)
=-\kappa\,\mathbb E_{\sigma,\epsilon}
[\ell_\theta(X;\sigma,\epsilon,c)-\ell_0(X;\sigma,\epsilon,c)],
\]

其中 ℓ 严格使用审计过的原生加噪分布、预条件化和坐标误差权重。κ 与 MSE 归一化、维度、σ权重有关，不能直接照搬图像库常数。

必须在文稿中明确：主实现的相对分数与同结果一致性，在该实现中是**变分/能量代理**，上述精确密度定理不会自动原封不动成立。这应通过小型精确模型与原生诊断分别验证，不可隐瞒。

### 11.3 具体实现要求

- 在真实 atom14 有效坐标上定义节点分数，避免将 winner/loser 不同数量的虚拟原子单独归一化后冒充统一概率。
- 首先确定固定、与 policy 参数无关的刚体坐标规范。依赖当前预测的 Kabsch 与不一致 mask 会改变代理解释。
- 每个节点的分数应独立于它与谁配对，避免 same sequence 在不同 pair 中使用不同标度。
- 先在每个构象内对若干 σ/noise 重复评估求平均，再比较相对分数；不能把随机噪声起伏当成条件几何差异。
- 同序列构象需有真实差异，且reference支持不能过低；同结果边必须与随机同分异序列边区分。显式边缘化扩展才需要额外记录importance权重的有效样本数。
- 对 policy=reference：a≈0、所有比较logit≈0、比较项≈log2；同结果边在初始处梯度为零。软标签在最优处的 BCE 不一定为零；不要套“loss必须到零”的验收规则。
- 同一 sequence 的几何约束不等同于把 denoiser 每一步输出钉住；也不能逐σ强行声称似然比不变。

### 11.4 合法几何补全是第一道可行性门槛

不能承诺这是几十行 loss 改动。现有权重版基本复用旧样本，新版需要正确的反事实几何和同序列多构象。工程成本中等，最难的不是优化器，而是 native representation 与合法局部几何实现。

第一步只挑约 32 对训练样本，覆盖双方正、双方负和背景翻转，做少量局部几何实现。若精确 hard decode、背景保持和合法性无法同时成立，不启动正式训练，更不能用不合法伪结构硬撑理论。

参考局部去噪并不自动等于 p0(X|S,c) 的精确采样。主实现因此只使用同序列局部点对约束，并将全局条件分布保持作为理想化分析；不要报告原生数据上的“精确条件KL”。

## 12. 最小判别实验

### 实验一：数据审计，已完成

复现符号分类、较优回退和截断前后稀疏性。随附脚本不调用任何模型。下一步可在新的训练条件和第二 reward 上复现同类统计，不能把这些训练对当成全体蛋白的普遍比例。

### 实验二：精确小模型

构建 2–8 个二值位点，每个序列搭配若干几何隐变量；显式定义参考分布、任意阶 reward 和精确 Gibbs 最优分布。比较：现有正权重、只保留符号的旧样本加权、有符号局部比较、完整 v2（局部比较加同结果边），以及知道精确目标的 oracle。

指标是序列边缘 KL、局部偏好符号正确率、条件几何 KL、目标分布误差，不是只看 reward。尤其必须复现本文两残基反例，确认旧 residual-shrinkage 在残差为零时仍不能表达背景翻转。

随附 `math_checks.json` 只检查恒等式和反例，没有模拟出新的蛋白收益，不能当作实验成功。

### 实验三：原生几何实现与代理校验

在约 32 对训练样本上获得合法局部编辑与 M=2 的同序列实现。记录 acceptance、失败类型、背景几何变化、decode parity、序列一致率以及密度代理的噪声方差。用 M=2/4/8 的小子集检查同结果边覆盖和噪声敏感性；这不是条件KL的精确估计。

不以任意固定 acceptance 阈值冒充生物标准；预先登记可承受的预算和失败率，并与用同样数据的全部基线共享过滤规则。

### 实验四：拆清楚“新数据”与“新算法”

最重要的对照是**所有方法使用完全相同的已查询序列、合法几何池、oracle预算和原生扩散计算预算**。

| 对照 | 回答的问题 |
|---|---|
| 当前 CF-DPO / Gain-only / Diff-only | 是否超过已有最强而不是最弱基线 |
| 反事实搜索，直接交付最高分，不训练 | 已支付的查询成本直接搜索能做到多少 |
| 新样本池 top-only diffusion FT | 是否只是多了更好的训练样本 |
| 新样本池普通/软标签 Diffusion-DPO | 是否只是构造了更干净的局部偏好对 |
| signed-only 局部比较 | 修正偏好方向本身的贡献 |
| signed局部比较 + 普通reference坐标anchor | 是否只是又加一个常见正则项 |
| 完整 v2：signed局部比较 + 同结果边 | 同序列相对密度一致性的额外贡献 |

不能只与旧192-pair N3比较新扩增池结果。若普通DPO在同池上已经追平，论文贡献就主要是反事实数据构造；不能继续包装成新的优化算法。

### 实验五：保留主奖励与独立验证

第一轮沿用现有24train/8dev，三种训练随机种子只是最低可行检查。Gain-only、Diff-only也要补多种子，否则无法判断小于训练方差的增益。

已被多轮查看和用来决定方向的 valid100，应保留为历史可比性集合，而不是反复称“全新未触碰测试集”。正式论文需要新增cluster隔离的盲测条件，先冻结再选择最终版本。

至少再验证一个独立 sequence reward；第二个蛋白设计场景更能支持泛化。主 reward 不变的实验里，Protenix-v2 与 Boltz-2 只用于独立诊断，不参与 reward，也不合并为一个总分。

## 13. 审稿人最强质疑与验收标准

| 质疑 | 不能回避的回答/实验 |
|---|---|
| 只是Shapley加权 | 不再输出单一重要性权重；用背景相关、可反向的局部比较直接监督 |
| 只是TokenRatio换成蛋白 | 关键新增是硬解码多对一结果的同序列相对密度约束；若该部分无效，创新优势大幅减弱 |
| 只是Di3PO式局部配对 | 必须在相同局部pair池上比普通DPO更好，不能把数据收益算作loss创新 |
| 多用了oracle/生成算力 | 报总queries、accepted/rejected补全、GPU时间、denoiser前反向次数；给等预算搜索和训练基线 |
| 理论只对精确似然成立 | 明确理想模型与BoltzGen变分代理差距，先做精确合成模型，再做原生代理校验 |
| 条件结构不变是否等于结构更好 | 不等于；不承诺总体RMSD提升，只检验有没有减少无奖励依据的额外变化 |
| 反事实不在数据流形上 | 几何lift是前置门槛；报告拒绝偏差与可行支持范围，不静默替换失败样本 |
| 分类器被过拟合 | 明确代理目标；新盲测、独立reward与refolder验证，不能声称湿实验功能提升 |

这个方案是否有较强的ICML竞争力，取决于“同池强基线之后仍有可重复的算法收益”，不是名字、公式数量或多加一个定理。

## 14. 明确停止条件

若合法几何补全不稳定，不继续大规模v2；保留原CF-DPO，并将可做的signed-only视为小幅修正，而非硬称顶会创新。

若同池普通DPO已达到完整v2，收缩贡献到反事实偏好纠错/数据构造，不能继续宣称新优化范式。

若同结果边只能降低自身代理误差，却不改善独立条件诊断，应视作代理失配，不增加更多正则系数把结果调漂亮。

若只在主classifier有收益而第二reward/新任务不成立，承认适用范围，不声称任意黑盒reward。

反之，若能同时显示：局部翻转场景中signed比较明显优于静态加权、同池下结果空间对齐优于普通DPO、独立测试中收益可复现，这比“CF-DPO再换一个权重”具有更明确的方法贡献。

## 15. 可用于论文的核心表述

> Global preference winners need not be locally preferred, and a decoded sequence admits multiple geometric realizations. We study preference alignment at the decoded-outcome level rather than treating counterfactual scores as nonnegative importance weights. Signed, context-matched interventions correct local preferences, while decoder-equivalence constraints on density ratios separate sequence reweighting from reward-irrelevant changes within the geometric realization of a fixed sequence.

上述为拟研究的问题与方法，不是已验证摘要。不能加入“显著改善结构”“对全部reward普适”“首次反事实DPO”等尚无证据的内容。

## 数据来源与复现

- **[D1]** 用户上传 `cfdpo_paper_stage_report.zip`，`docs/PAPER_STAGE_RESULTS.md`，重点第3–8、10、13–15节。是报告陈述，不是对全部代码的重新验证。
- **[D2]** 用户上传 `next_stage_credit_report.zip`，`data/counterfactual/residue_credit.csv`，2,996行。
- **[D3]** 同包 `data/counterfactual/pair_credit_summary.csv`，192对。
- **[D4]** 本次新增复算：`audit_and_math_checks.py`、`signed_credit_audit.json`、`signed_credit_counts.csv`、`per_pair_reaudit.csv`、`math_checks.json`。输入SHA256写入JSON。

复现命令：

```bash
python audit_and_math_checks.py \
  --credit-csv source_data/residue_credit.csv \
  --pair-csv source_data/pair_credit_summary.csv \
  --outdir reproduced
```

## 外部参考文献

外部文献用于界定已有方法和可借鉴机制，不代表已证明本文v2的新颖性或效果。检索没有发现等同完整机制，不等于穷尽全部文献。

- **[R1]** Xue et al. *Improving Protein Sequence Design through Designability Preference Optimization*. arXiv:2506.00297, 2025. https://arxiv.org/html/2506.00297v1
- **[R2]** *Decomposed Direct Preference Optimization for Structure-Based Drug Design*. arXiv:2407.13981. https://arxiv.org/html/2407.13981v2
- **[R3]** *Beyond Uniform Credit: Causal Credit Assignment for Policy Optimization*. arXiv:2602.09331, 2026. https://arxiv.org/html/2602.09331v1
- **[R4]** Nguyen et al. *TokenRatio: Principled Token-Level Preference Optimization via Ratio Matching*. arXiv:2605.12288, 2026. https://arxiv.org/html/2605.12288v2
- **[R5]** Reddy et al. *Di3PO — Diptych Diffusion DPO for Targeted Improvements in Image Generation*. arXiv:2602.06355, 2026. https://arxiv.org/html/2602.06355v1
- **[R6]** Mayer and Wüthrich. *Shapley Values: Paired-Sampling Approximations*. arXiv:2508.12947, 2025. https://arxiv.org/abs/2508.12947
- **[R7]** Rafailov et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html
- **[R8]** Wallace et al. *Diffusion Model Alignment Using Direct Preference Optimization*. CVPR 2024; arXiv:2311.12908. https://arxiv.org/abs/2311.12908
- **[R9]** BoltzGen official implementation, reference commit `a3149cf18eeb58648d1abbb27539bd73f746cdda`; `data/feature/featurizer.py` and `data/const.py`. https://github.com/HannesStark/boltzgen/tree/a3149cf18eeb58648d1abbb27539bd73f746cdda
- **[R10]** Lugmayr et al. *RePaint: Inpainting Using Denoising Diffusion Probabilistic Models*. CVPR 2022. https://openaccess.thecvf.com/content/CVPR2022/html/Lugmayr_RePaint_Inpainting_Using_Denoising_Diffusion_Probabilistic_Models_CVPR_2022_paper.html
