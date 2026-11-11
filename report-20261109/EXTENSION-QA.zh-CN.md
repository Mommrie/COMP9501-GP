# 强化实验：教授最可能追问的机制问题

数值口答以最终新增结果表为准；以下解释固定设计，不预设方法获胜。

## 1. 为什么又加FP32 CE？

> The new response controls use FP32 to reduce numerical ambiguity in small hidden deltas. We include a CE baseline at the same precision and capacity, so a change relative to the earlier BF16 run is not mistaken for an activation effect.

新hidden与旧BF16 CE不是只差目标函数；机制对照应在新FP32组内做。新组均L4，历史A100结果单列。

## 2. 为什么需要打乱目标，CE对照还不够吗？

> CE controls capacity and supervised optimization. The shuffled-response control keeps the same teacher-target set but breaks input correspondence. It asks whether correct pairing matters beyond adding a response-distribution regularizer.

它是机制控制，不是拿故意较差的算法当竞争者。必须同时看真实目标vs CE、真实vs shuffle；只胜shuffle却输CE，不能说整体更有用。

## 3. 减base是不是提出了新距离？

> Subtracting the same base response cancels in the squared-error numerator. Our explicit design choice is normalization by the teacher-update energy, with a fixed floor. We do not present this identity as a new metric.

这是值得主动讲清的数学事实。floor=1e-4、hidden系数=1是固定设计，不是理论最优值。

## 4. hidden loss下降而accuracy下降，是否实现错了？

> It can reflect a mismatch between the surrogate and the task. We match a masked token-mean representation, while classification uses the model's native pooling and task head. Numerical parity and gradient checks validate the implementation, not the assumption that this surrogate preserves decisions.

先检查代码等价、冻结梯度、数据权限和数值，再解释代理目标局限。不要遇到负结果直接换指标或删掉该profile。

## 5. 全局5参数vs逐层97参数是否公平？

> It is a capacity and layer-flexibility ablation, not an equal-capacity comparison. The equal-capacity mechanism test is among the ninety-seven-parameter CE, aligned-response and shuffled-response variants.

它不能单独证明某层有特定语义，也不等于已经重训q-only、v-only或只适配后半层的teacher。

## 6. Router已经知道task，为什么还需要学习？

> The defined question is whether a different backbone expert helps an individual input under the same known task head. Task-ID selection is therefore the required direct baseline. We are not claiming unknown-task identification.

expert router的task one-hot使用的是当前协议本来就提供的信息。source router只是来源识别参照，不是主要部署优势。

## 7. Best-expert标签从哪里来？

> We evaluate every expert on fit examples using the true task head and choose the one with minimum ground-truth negative log-likelihood. This uses fit class labels and multiple offline expert passes, which we include in the information and computation budget.

不能称data-free、无标签或低成本拟合。不同head的0/1语义不同，所以始终固定当前任务head后比较backbone专家。

## 8. Oracle为什么可以放进结果表？

> It is an analysis-only accuracy ceiling for the frozen expert bank. Final labels identify whether any expert can correct an own-expert error. This quantifies headroom; the oracle is not deployable and does not prove that our features can predict those choices.

准确率上界不自动是MCC或native宏均值的最优上界。图表必须显著标注final GT，不能把oracle当方法优势。

## 9. 高oracle上界为何实际router未提升？

> Complementary errors need not be predictable from the available frozen features and small fit set. We report how often routing helps and harms the own expert. Their difference equals the accuracy change and explains whether the policy realizes that headroom.

“存在更好的选择”与“能用可获得信息学出选择”是两个不同命题。Oracle身份agreement也不等于任务准确率，因为多个expert可能都预测正确。

## 10. 新增实验还算独立测试吗？

> The new teachers and fit/select partitions remain separated. However, these extensions were designed after inspecting v3 final scores, so the evaluation is exploratory. Locking the next selection file does not erase prior research exposure.

这是本轮必须保留的限制，不因为结果好就删去。三seed仍共用一个数据划分。

## 11. 为什么新结果用L4，成本图却是A100？

> A100 was unavailable for the extension, so all new matched controls ran on the same L4 in FP32. The A100 timing is a separate historical implementation benchmark. We do not combine those stages into a jointly measured accuracy-latency Pareto claim.

额外base pass和expert分组是结构性成本，旧毫秒数仅支持那次A100实现。新expert router的L4部署耗时若未测，就直接说明未测。

## 12. 这么多负结果，最后怎么总结？

> We distinguish implementation success from hypothesis support. Learned static mixtures can improve the fixed weight-space grid in this case study, while matched controls determine whether the proposed functional signal adds anything beyond supervision, capacity or target distribution. We report the measured outcome and the conditions under which the original expectation fails.

如果新对照真的有一致正结果，也只能在这套固定设置里报告；没有运行现代完整对齐基线、更多模型和独立外部目标，不能升级为SOTA结论。

## 选文与Part1的直接桥接（新增）

新增两页PPT安排、完整三seed结果、代码与论文公式对应、英文口述及12组追问见 [CSA桥接材料](CSA-BRIDGE-RESULTS.zh-CN.md)。这部分直接应用CSA压缩expert响应，原合并主线结论保持原样；response compression不等于weight compression或推理加速。
