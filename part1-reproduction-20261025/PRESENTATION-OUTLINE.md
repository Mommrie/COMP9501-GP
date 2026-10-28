# CSA Part 1：12 分钟正文讲解稿与问答准备

工作稿。按约 12 分钟正文 + 3 分钟问答、合计 15 分钟准备；讲义未另列问答时长，具体分配和 Group 13 的实际场次仍待核对。成员尚未分配，以下按内容而非人名组织。

2026-10-28 新增 [代码与展示证据包](presentation-20261028/README.md)，提供图注、原文证据、存储预算实验与讲解路线；本提纲供总体结构参考。

原文：[Xu et al., 2008](https://doi.org/10.1109/TCSVT.2007.903317)。下列页码为 **PDF 页码**（期刊页码 = PDF 页码 + 35）。课件背景：Lec4 PCA/LDA（课程附件不随仓库分发）、Lec5（课程附件不随仓库分发）。这是讲稿草案，不是已经提交的 slides。

## 1 — Can structure help us learn from few images?（45 秒）

画一张矩阵图像与其逐行排列成向量的对照，提出问题：同一份像素，学习投影时的结构约束会改变什么？不要说“向量化把像素信息丢掉”。向量化本身可逆，区别在模型的可行子空间与参数规模。

开场英文：*The question is not whether reshaping destroys pixels. It is whether a structured projection is easier to estimate when examples are scarce.*

## 2 — PCA as reconstruction（60 秒）

中心化向量 x∈R^D，W∈R^(D×d)，WᵀW=I：

`min Σ ||x − WWᵀx||²  ⇔  max Σ ||Wᵀx||²`。

用“保留的能量”和“丢失的能量”两块区域画图；强调 PCA 保留最大训练方差，优化目标不是分类准确率。依据原文 p3 / Lec4。

## 3 — Two small projections, one compact core（60 秒）

先讲二维：X∈R^(H×W)，U∈R^(H×r₁)，V∈R^(W×r₂)，两者列正交。

`Y = Uᵀ X V`, `X̂ = U Y Vᵀ`。

以 14×14 → 4×4 → 14×14 的尺寸箭头说明行、列方向各自投影；不要一开始就铺满 mode-k 符号。依据原文 p3–4 Fig.1。

## 4 — Generalize the object, not the training set（60 秒）

三维示例：一张图像的高×宽×Gabor filter 响应。每个 mode 一个投影，投影后仍是小 tensor core。目标是在正交约束下最小化所有训练对象的重建误差。

区分“一张样本是一个张量”和“把多人、多姿态组织成一个大张量”；这为后面 Tensorface 的区别埋伏笔。原文 p4–5 Fig.2 / Fig.3。

## 5 — One conditional update is an eigensystem（90 秒）

二维推导只保留三行：

1. `L(U,V)=Σ||X||² − Σ||UᵀXV||²`，第一项固定。
2. 固定 V，令 `B=Σ(XV)(XV)ᵀ`，最大化 `tr(UᵀBU)`。
3. U 取 B 的前 r₁ 个特征向量；固定新 U，类似更新 V。

关键口头解释：先压缩其他方向，再对当前方向的条件协方差做 PCA。特征分解解决的是这个子问题，不是一次性解决所有投影的联合最优。原文 Theorem 1 p4，证明 p10。

## 6 — Alternating optimization: what is guaranteed?（60 秒）

流程：初始化 → 更新 U → 更新 V → 检查目标变化 → 循环。每次条件最优更新使 captured energy 不减，等价于 reconstruction error 不增；目标有界。

展示 [本次能量曲线](artifacts/csa-convergence-2026-10-25.png)，标题明确“our 2D illustration”。这证明数值运行符合单调性，不证明全局最优或投影矩阵唯一收敛。实现按相对目标变化≤1e−7 停止，最多 100 sweeps。2026-10-28 视觉复核：原文 Procedure 1 的相似度停止条件为 `> 1−epsilon`，旧“方向矛盾”表述撤回。我们的目标变化规则与原文不同，须明确说明。

## 7 — CSA is constrained PCA（75 秒）

向量化后的投影是 Kronecker 积。使用列优先 vec 时写 `vec(UᵀXV)=(Vᵀ⊗Uᵀ)vec(X)`；实际 NumPy C-order 使用另一对应顺序，代码已有数值恒等式检查。

例子：14×14，core=4×4。PCA basis=196×16，CSA basis=14×4+14×4。前者更灵活，后者更紧凑。**相同 latent 维数不等于相同总存储。** 原文 Claim 1–2 p6。

## 8 — Where do 2DPCA and Tensorface fit?（60 秒）

用简表：PCA 学向量投影；2DPCA 只压缩一个方向；CSA 同时压缩多个 mode；Tensorface 用姿态/光照等外部因素组织 ensemble。原文 p5–6。

避免“Tensorface 和 CSA 只是同一算法的名字不同”。它们可能共享张量分解工具，但样本组织、因子含义和数据需求不同。展开产生更多相关列，不会凭空创造独立训练样本。

## 9 — Read the original results with the right budget（75 秒）

只选原文两组证据：

- p7 Fig.5：ORL / PIE-1，三维 Gabor 表示，RMSE 对 compression ratio，计入投影矩阵存储。
- p8–9 的 ORL / MNIST 识别：维数、表示及后续 LDA 设置不同；最近邻分类，作者报告最佳配置。

不口述未经核对的精度数字，也不称“同维数 CSA 训练重建必然胜 PCA”。原文最佳配置的选择协议未明确给出独立 validation，现代复现应说明自己的选择集。正式 slides 如引用原图需标清图号与出处，仅用于课程私下学习。

## 10 — Our controlled illustration（75 秒）

展示 [比较图](artifacts/csa-comparison-2026-10-25.png)。三训练子集 seeds；每类 1/3/10/30 张；灰度 14×14；相同 core=4 或 16；固定 1-NN 与 native held-out test；没有 Gabor 或 LDA。

选一组便于读图：每类 30 张、latent=16，CSA 80.2±1.4%、PCA 83.7±0.3%；CSA basis 112 个数、PCA 3,136 个数（共同均值另计）。讲结果原貌：PCA 这组识别更高，CSA basis 更小。这不是原论文表格复现，也不据此否定原文不同预算/表示的结论。

## 11 — What the method buys, and what it assumes（45 秒）

结构约束降低 basis 存储和条件协方差尺寸，但可能限制可表达子空间；训练误差更小不必然识别更好，测试结果还依赖数据规模、表示、维数与选择协议。迭代次数和计算布局决定实际速度，不能从矩阵尺寸直接声称现代实现固定加速倍数。

## 12 — Takeaways（15 秒）

结束句：*CSA learns compact multilinear projections by alternating conditional PCA steps. Its value depends on the structure and the comparison budget; neither optimal training reconstruction nor universal recognition superiority is guaranteed.*

正文约 12 分钟；最后一页保留公式和 Q&A 提示。完整 [补充结果](RESULTS.md) 与 [代码](code/README.md) 可放附录。

## 备用问答

| 可能问题 | 建议回答 |
|---|---|
| 为什么不直接 PCA？ | 若只看固定训练集与相同 latent 的最优线性重建，PCA 可行域更大。CSA 的动机是结构、存储和有限样本估计；收益取决于数据与预算。 |
| CSA 是有监督的吗？ | 本文基本重建准则无监督；CSA+LDA 的后续 LDA 有监督。本次降维不使用类别标签。 |
| CSA 的“最优”是什么？ | 固定其他 mode 的条件子问题可由特征分解最优求解；联合非凸问题没有全局最优保证。 |
| 比较同一个 core 大小公平吗？ | 它隔离 latent 大小，但总 basis 存储不等。另一问题是匹配总 compression ratio，两种比较应分别说明。 |
| 很少图像，展开后样本变多是否解决一切？ | 行列向量仍来自同一批图像且相关，结构先验可能有帮助，不能视为增加同等数量独立观测。 |
| 随机初始化会怎样？ | 可能收敛到不同驻点。本次只用一种确定性行列协方差初始化，未进行多起点最优性研究。 |
| 为什么我们的 CSA 没胜 PCA？ | 数据、表示、维数和预算与原文不同；本次是灰度二维、小 core 设置。保留负结果比测试挑配置更有信息。 |
| 可以用于深度网络吗？ | 可作为输入表示或特征压缩工具，但本展示解释 2008 年线性张量子空间方法，不把现代变体当作原文贡献。 |

## 组员审阅时要完成

核实实际场次与 15 分钟要求；按组员专长分配讲解；选择合法可分享的示意图或自行重画；核对引用原图/表数字；练习一次计时与交接。AI 协助推导检查、代码与讲稿草案，成员需独立理解和核实；COMP9501 具体 AI 披露要求仍待课程确认。
