# 原文证据与实现边界

入口：[材料包](README.md) · [公式与代码](CODE-WALKTHROUGH.zh-CN.md)

原文：Dong Xu et al., *Reconstruction and Recognition of Tensor-Based Objects With Concurrent Subspaces Analysis*, IEEE TCSVT 18(1):36–47, 2008，DOI [10.1109/TCSVT.2007.903317](https://doi.org/10.1109/TCSVT.2007.903317)。下文 **PDF 页码从 1 开始，期刊页码 = PDF 页码 + 35**。2026-10-28 读取方法/实验相关页，并视觉核对 PDF p4、p7、p9 的公式、算法框和 Table IV；不把早期导读中的概括当作原始证据。

本共享仓库不附带带图书馆授权信息的原 PDF、截图或课程附件。组员可通过学校/出版方的授权途径查原件。本包的示意图和表格重绘由我们制作，并标明来源。

## 证据地图

| PPT / 讲解要点 | 论文位置 | 我们能展示的对应物 | 边界 |
|---|---|---|---|
| 向量、矩阵和高阶张量的降维/重建 | PDF p3，Fig.1，Eq.(1)–(4) | 01 流程、02 重建图 | 向量化可逆，不应说它丢掉像素；模型结构约束不同。 |
| 最优重建准则 | PDF p4，Eq.(5) | code 中重建残差与 captured energy | 优化目标是训练重建，不是识别 accuracy。 |
| 固定其他 mode 后做特征分解 | PDF p4，Theorem 1、Eq.(6)、Fig.2 | 二维 einsum 与高阶 partial/unfold 实现 | 条件最优，不等于一次求出全局最优。 |
| 依次使用已更新的投影矩阵 | PDF p4，Procedure 1 | `fit` 的 mode 循环 | 使用顺序更新，不是把所有更新都建立在旧 factors 上。 |
| 目标函数单调与有界 | PDF p5，Eq.(7) | 07 能量曲线、数值检查 | 目标收敛本身不证明投影唯一或全局最优。 |
| CSA 是带 Kronecker 约束的 PCA | PDF p6，Claim 1–2 | 03 basis 图、Kronecker 恒等式测试、1D PCA 等价测试 | 相同 latent 的训练重建有 PCA 下界。 |
| 二维 CSA 与其他矩阵方法的关系 | PDF p6，Claim 3、§III-C | 现有二维代码；讲解中说明与 2DPCA 的联系 | 本轮不声称复现了所有相关方法或 LDA。 |
| CSA 与 Tensorface 的对象组织不同 | PDF p5–6，Fig.3、§III-A | 一张样本为一个张量，N 为 batch 的解释 | 不是把整个数据库的类别/姿态轴当作样本内部 mode。 |
| 原文收敛图 | PDF p7，Fig.4 | 我们的 07 能量曲线另行标注 | 原图纵轴是相邻投影的相似度；我们的纵轴是目标能量。 |
| 原文重建比较 | PDF p7–8，Fig.5、§IV-B | 05/06 存储预算图作机制说明 | 原图使用 ORL/PIE 三维 Gabor 和含 basis 的 compression ratio；不是同 latent。 |
| 原文 MNIST 识别结果 | PDF p9，Table IV、§IV-C | 10 原表重绘 | 属作者结果，配置/数据/表示与我们不同。 |
| 特征、维数和分布会影响结果 | PDF p9–10，Discussions | 04/05 新实验与负结果解释 | 不把一种小样本设置外推为普遍优势。 |

## Table IV：已核对的原文数字

下表按 PDF p9 / 期刊 p44 的表格视觉抄录，保留作者报告的最佳配置。`Gm/Pn` 表示每个数字 m 张 gallery/training、n 张 probe/test；不是 MNIST 原生 train/test 划分。

| 原文方法 | G30 / P470 accuracy (%) | 最佳降维配置 | G40 / P460 accuracy (%) | 最佳降维配置 |
|---|---:|---|---:|---|
| PCA (Gabor) | 88.5 | 60 | 90.5 | 50 |
| CSA-3 | 92.8 | 6×9×18 | 92.4 | 15×5×13 |

设置：原文描述使用 MNIST native test 的前半部分，共 5,000 张、每数字 500 张；输入是 28×28×40 的 Gabor 特征（五尺度、八方向），降维后用最近邻。这里记录的是**原文描述**；我们没有取得作者的具体样本 ID 来验证其实际平衡取样方式，也没有重跑其代码。

作者在 PDF p8 §IV-C 说明报告最佳结果及配置；所读实验说明没有给出独立 validation 的完整选择程序，选择协议（待核对）。因此 10 图可说明“作者在这些设置下报告 CSA-3 更高”，不能说明同维度、同存储、同调参预算下我们也验证了相同提升。CSA core 的维数乘积远大于 PCA 所列 dimension，此处应保留配置，避免把这张表当成严格同维数控制。

## 原文 Fig.5 的存储口径

原文 PDF p7 明确把投影矩阵算入存储。设 N 个三阶样本大小 `m1×m2×m3`，CSA 各 mode rank 同为 r：

$$s_{PCA}=d(N+m_1m_2m_3),\qquad
s_{CSA}=Nr^3+r(m_1+m_2+m_3),\qquad
CR=Nm_1m_2m_3/s.$$

本包二维演示对应为：`s_CSA=N*r²+(H+W)*r`，`s_PCA=N*d+H*W*d`。我们的新预算检查**额外显式计入共同均值 H*W**，固定 N=300；没有复制原文图的数值或声称做了其完整实验。

原文 Fig.5 的 RMSE 是每图重建平方范数的平均再开方。对于本包固定 D=196：原文形式的 RMSE = `sqrt(D * pixel_MSE)`。图像尺度、表示和归一化也不同，不能把其纵轴与我们的 pixel MSE 直接比大小。

## 本次纠正的旧表述

1. **停止不等号**：PDF p4 Procedure 1 的投影相似度条件清楚印为 `> (1−epsilon)`；旧说明“印刷方向矛盾”撤回。实现使用目标变化，是我们自己的明确约定。
2. **收敛保证**：论文写到局部最优；我们的数值检查直接支持的是目标单调且有界。不能从一张曲线证明全局最优，也不能忽略退化特征值/非唯一子空间。
3. **USPS**：旧冻结 plan.scope 的 USPS 为文本笔误，原文实验是 FERET、ORL、CMU PIE 和 MNIST；新材料采用正确名称，旧执行方案保留供核验。
4. **小样本论证**：展开使列向量数量变多，但它们有相关性；这不是增加同等数量独立训练图像。
5. **速度论证**：原文复杂度分析有样本数/维度假设；我们的 PCA 使用 SVD，没有机械地建立巨大协方差。未做受控 timing benchmark，不声称固定倍数加速。

## 我们与原文的差异清单

| 维度 | 原文相应实验 | 我们的 MNIST 演示 |
|---|---|---|
| 数据 | 包含人脸；MNIST 使用 native test 的一部分再分 gallery/probe | native train 取训练子集，native test 固定 1,000 张 |
| 特征 | 灰度或 40 通道 Gabor，MNIST 为三维 Gabor | 28×28 灰度均值下采样为 14×14 |
| 方法 | CSA-2 / CSA-3，部分比较含 LDA | 二维 CSA、PCA；高阶另用合成数据验证 |
| 选择 | 原文报告各方法最佳配置 | 全部预列 rank 报告，不根据 test 选最佳 |
| stopping | 相邻投影相似度，epsilon=0.001 | 完整 sweep 的相对目标变化，tol=1e-7 |
| 比较预算 | 重建图匹配 compression ratio；识别表报最佳配置 | 同 latent 与存储约束两种问题分开报告 |

可以称“依据原文重建准则的可解释实现和受控补充实验”；不能称“已严格复现原论文所有表格”或“已验证任意数据上 CSA 优于 PCA”。
