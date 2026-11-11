# 从 CSA 论文到 LoRA：直接迁移实验协议

2026-11-11，在计算前固定 [配置](../configs/csa-bridge-v1.json)。此前的结果已经看过，本实验是探索性扩展，不称新盲测。原有合并实验保持原样。

## 论文、Part 1、Part 2 的对应

选文为 Xu et al., *Reconstruction and Recognition of Tensor-Based Objects With Concurrent Subspaces Analysis*, TCSVT 2008，DOI [10.1109/TCSVT.2007.903317](https://doi.org/10.1109/TCSVT.2007.903317)；[作者机构页面](https://www.microsoft.com/en-us/research/publication/reconstruction-recognition-tensor-based-objects-concurrent-subspaces-analysis/)。原文 Eq.(4) 给投影/重建，Eq.(5) 给重建准则，Eq.(6)、Theorem 1、Procedure 1 给条件子空间更新；Claim 1 讨论 Kronecker 结构约束。Part 1 的 MNIST 实现解释上述机制，本实验直接导入同一个 [TensorCSA](../part1-reproduction-20261025/code/csa_tensor.py)，源码 SHA 随结果记录。

每个输入 x 形成 `X(x)[expert, feature]`：四个任务 adapter 的最后层 masked-mean hidden response，分别减去同一输入的 base response。矩阵两个轴分别是真实 expert 身份和隐藏坐标，不把一个向量任意切成“图像”，样本轴也不作为 CSA mode。

对中心化矩阵 Z=X−M，学习 U∈R^(4×rE)、V∈R^(H×rH)，使 Σ||Z−UUᵀZVVᵀ||² 最小；core=UᵀZV。沿 expert/feature 两个 mode 交替做条件特征空间更新。原文的重建数学直接沿用，数据对象从图像变为 LoRA 功能响应。中心化、协方差初始化、能量停止准则属于我们已公开的实现约定；最多30 sweeps，不保证全局最优。

## 两种公平性口径

- 相同 latent：PCA rank d=rE·rH。PCA 在训练集上有更自由的投影，训练重建误差应不高于 CSA；这是一项自动检查，不预设 CSA 胜出。
- 相同存储上限：固定 N=512 条缓存，以相同精度的 scalar 数计入 mean、basis、codes。CSA 为 EH+E·rE+H·rH+N·rE·rH，PCA 为 EH+EH·d+N·d；取不超 CSA 预算的最大整数 d。预算取整剩余明确保留。实际总缓存编码成本还包括序列化、logits、router、adapter/base 等，本比较不包含它们，不换算成端到端部署节省。

全部表示以 float64 计算与缓存，因此 scalar 比率对应同精度数组字节比率；不把该比率当 ZIP 压缩比。basis 在512条 coefficient_fit 上无标签学习，selection/final 不参与表示拟合。每 seed 使用相同冻结任务银行和五分区协议。rank 网格全部报告，不通过 final 挑“最佳设置”。

## 下游信息保留检查

在同一 fit 上以最小 GT-NLL expert 为监督目标；对 CSA/PCA core、raw delta 和 base features 使用同样的 StandardScaler、已知任务 one-hot 与 logistic router。C=.1/1/10 只在 coefficient_selection 的任务 native score 上选择，平分取小 C。锁定全部选择后，才收集原内部 final 响应并评估。task-ID 是可用参照，GT oracle 是离线二分类 accuracy ceiling。

重建误差除以关于 fit mean 的中心化能量；训练、selection 和内部 final 分开报告。下游指标按 SST2/RTE accuracy、MRPC (accuracy+F1)/2、CoLA MCC 宏平均，不等于平均 accuracy。三 seed 是原有三训练重复，新增表示学习不增加独立 teacher seed。

## 实际可主张的范围

这是 CSA 在现代 LoRA 响应上的真实应用和结构化缓存研究，不是用 CSA 压缩 LoRA 权重，也不是对 merge coefficients 做 CSA 更新。生成表示需执行全部四个 expert；不能把离线路由分数解释为可部署的加速。如果缓存重建很好而下游任务仍差，这恰好说明重建保真与任务效用需要分别验证。无论正负结果，均保留完整 rank 网格及相同 latent/预算 PCA 对照。

## 复现

```bash
python -m unittest discover -s code -p test_csa_bridge.py
python code/csa_bridge.py --study outputs/independent-v3-20261109 \
  --previous outputs/activation-sensitivity-20261111 \
  --plan configs/csa-bridge-v1.json --output outputs/new-csa-bridge
```

需要已恢复的 matching teacher bank、原 sensitivity 的三角色缓存及锁定配置；输出目录必须全新。模型计算和矩阵分解在 Colab；本地只负责源码、传输和 SHA/CRC 核验。
