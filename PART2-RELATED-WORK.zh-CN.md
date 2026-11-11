# Part 2 相关工作与贡献边界

2026-11-09 核对。仅整理与本项目直接相关的方法段落，不称逐篇完整复现。原论文数字不与我们的四任务结果横向比较；模型、任务和数据权限不同。

## AdaMerging：逐层系数并不是我们的新发明

- **问题与方法**：给 task vector 学习任务级或层级系数，冻结原模型；用无标签多任务测试输入的预测熵优化系数。见原文 §3.2，尤其 §3.2.1–3.2.2。[论文全文](https://arxiv.org/html/2310.02575)
- **与本项目的联系**：我们也学逐层混合权重，但用 softmax、共享 gain 和四个 LoRA ΔW，比较 CE 与 teacher-logit KL；拟合与选择数据来自独立源 train 分区，最终评估不做 test-time adaptation。这是实现与实验设定差异，不能单凭差异声称新算法。
- **局限与复现范围**：本项目没有按该论文的视觉模型、熵目标和测试适应协议复现 AdaMerging，也没有与其官方完整实现实测，不能声称胜过 AdaMerging。
- **阅读范围**：核对 §3.2；完整实验和全部附录未逐项复现。

## RegMean：使用 activation 统计指导合并已有先例

- **问题与方法**：用线性回归让合并层输出接近各原模型；基于层输入的 Gram 统计解出合并权重，Transformer 各线性层分别处理，并收缩 Gram 的非对角项。见 §3.2–3.4、Algorithm 1。[论文全文，v6](https://arxiv.org/html/2212.09849v6)
- **与本项目的联系**：都关注功能响应而非只看权重距离；但旧局部响应原型不是 RegMean 的完整层权重闭式解，新 KL 原型匹配的是分类输出。不能把“activation-aware”这个名字本身当创新。
- **局限与复现范围**：原方法需要输入统计，不能把 dataless 理解为计算统计也不接触数据；我们的独立分区和 CE/KL 对照没有替代 RegMean 官方复现。本轮没有 RegMean 基线成绩。
- **阅读范围**：核对方法与统计要求；没有复现原实验组合。

## KnOTS：LoRA 合并中的子空间与对齐问题已有直接研究

- **问题与方法**：分析 LoRA 更新的表征对齐，采用 CKA 作诊断；逐层拼接任务更新做共享 SVD，再在对齐的任务分量上应用合并规则。见 §3、§4、Appendix B。[论文全文](https://arxiv.org/html/2410.19735)
- **与本项目的联系**：原文也研究 LoRA 功能对齐。我们的 cosine、最终层池化 Δactivation 与其跨层 CKA 定义不同；旧六任务对相关性不能算 KnOTS 的严格复现，更不能泛化为对齐已解决。
- **局限与复现范围**：本项目没有实现 KnOTS 的共享 SVD 变换，也没有复现其任务头/输出协议；不把旧权重空间基线叫 KnOTS。本项目使用已知 task ID，无法由此声称未知任务通用模型能力。
- **阅读范围**：核对对齐动机、SVD 方法及 CKA 附录；未复现其完整 benchmark。

## 展示时如何准确定位我们做的工作

可以说：我们实现了一个小模型、多任务 LoRA 合并研究流程，在固定数据预算下比较功能输出监督与标签监督，并诊断相似度、数据权限、来源线索和部署成本。逐层混合、蒸馏、相似度诊断分别有先例；课程项目的贡献主要是明确假设、实现对照、报告证据和失败条件，尚无依据宣称 SOTA 或首创。

Part 1 的 CSA 提供“在结构约束下选择子空间、明确存储预算”的理解背景。原有 Part 2 合并器采用24层四任务混合 logits 加一个gain，没有调用CSA投影更新LoRA。11月11日新增响应压缩分支后，两部分已有直接算法联系：同一个TensorCSA作用于expert×feature响应矩阵；尚未证明该压缩可改善LoRA权重合并，也不把响应压缩称为LLM权重压缩。详见文末桥接结果。

英文口答：

> Layer-wise merging and functional matching already have established precedents. Our project studies a narrower empirical question: whether teacher-output matching helps under a fixed LoRA merging capacity and a separated data protocol. We also examine when similarity and routing evidence can be misleading. We distinguish our implementation choices from algorithmic novelty, and we report the outcomes even when the original hypothesis is not supported.

## 2026-11-11：从概念联系推进到直接CSA应用

原有LoRA合并主线没有执行CSA权重投影；本轮新增 [CSA功能响应桥接](report-20261109/CSA-BRIDGE-RESULTS.zh-CN.md)，直接复用Part1 TensorCSA到expert×hidden response矩阵。三seed、七rank、等latent/计basis预算PCA、下游离线expert选择及重载审计完成。代表CSA2×8用148.1 KiB，PCA16用540.0 KiB，PCA3用124.0 KiB；final relative squared error分别0.4848/0.3843/0.5785。它建立算法与实验上的联系，但不称CSA权重合并、无标签路由或推理加速。整合报告17页，32项云端测试通过。
