# Task-conditioned functional matching

[中文使用指南](README.zh-CN.md)

本轮探索把“cosine 越高越安全”的不稳定判据，改成线性层输出匹配：在各 expert 自己任务的 calibration 输入上，估计所有 LoRA 的线性响应 Gram，求 simplex 上的全局或逐层系数。它与 RegMean 的输出匹配思想相关，是受限于 adapter 张成空间的工程研究原型；不声称算法首创。

统一输入、同权重的平方响应匹配会退化为 uniform average，这一事实作为解析/单元控制。真正新增的信息是各任务自己的输入分布。gain 的选择与 uniform 对照一致，避免把幅度调整误记为 activation-aware 的贡献。

`maths.py` 为 simplex 二次优化；`run.py` 为无标签响应收集、calibration 参数选择和锁定后的 validation。方案见 [functional-v1.json](../../configs/functional-v1.json)。所有模型使用原任务头。原三 seed 银行不覆盖。
