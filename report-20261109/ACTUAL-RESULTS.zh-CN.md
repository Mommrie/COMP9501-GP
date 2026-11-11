# Part 2 最终结果精讲：2026-11-11 云端完整复核

## 可以讲出的核心结论

原 proposal 的三条假设已经分别落到实现和对照上。v3 的学习系数改善当前固定权重基线（CE 0.7132、TIES 0.6146、Average 0.5578），但 individual 仍为 0.7695。输出 KL 为 0.6987，平均低于同容量 CE。新实验进一步发现：直接 hidden 目标也没有带来超出同精度 CE 的收益；任务性能导向 router 尚不能实现 expert bank 的 oracle 潜力。项目贡献应围绕受控实现、可复现协议和这些失败条件来讲。

本页新结果来自 11 月 11 日 Colab **L4、FP32、seeds 42/43/44**，完整预测在本次归档内。它是对冻结扩展的恢复重跑，复用同12个teacher，**不是六次独立重复**。原始五分区、教师训练与主 CE/KL 研究见 [v3 报告](../PART2-RESULTS-V3-2026-11-09.zh-CN.md)。扩展设计前已看过 v3 final，结果仍属探索性。

## 1. 直接 activation 控制

| 方法 | Native 主指标 | Mean accuracy | Worst-task accuracy |
|---|---:|---:|---:|
| CE FP32（97参数） | 0.7130 ± 0.0211 | 0.7861 ± 0.0029 | 0.6901 ± 0.0137 |
| Hidden-only（97） | 0.5901 ± 0.0259 | 0.7285 ± 0.0010 | 0.5820 ± 0.0170 |
| CE + hidden（97） | 0.6397 ± 0.0281 | 0.7643 ± 0.0182 | 0.6927 ± 0.0407 |
| CE + shuffled（97） | 0.6195 ± 0.0056 | 0.7510 ± 0.0195 | 0.6641 ± 0.0629 |
| Hidden-only（5） | 0.5841 ± 0.0297 | 0.7298 ± 0.0166 | 0.6055 ± 0.0768 |

CE+hidden − CE 的配对 native 差值均值 **-0.073280**，仅 0/3 seed 为正；CE+hidden − shuffled 为 **+0.020254**，2/3 为正。真实配对有时优于打乱目标，但这没有转化为相对 CE 的额外任务收益。不能只展示 real vs shuffled 而删掉 CE。逐层 hidden − global hidden 均值 +0.006011，这是5与97参数的容量消融，不是同容量机制检验。

| Seed | CE+hidden − CE | CE+hidden − shuffled | Layer hidden − global |
|---|---:|---:|---:|
| 42 | -0.054615 | +0.013141 | +0.007953 |
| 43 | -0.067016 | +0.053795 | +0.008625 |
| 44 | -0.098211 | -0.006175 | +0.001454 |

选中 checkpoint 的 aligned hidden-fit MSE，按四任务/三seed平均：CE 2.0763；hidden-only 1.0941；CE+hidden 0.4871。这些是校准区诊断，不能替代 final 分类指标。我们匹配末层 masked token mean，分类使用原生 pooling/head；较低表示误差并不保证更准确的决策。

**英文口答：** The aligned hidden objective changes the representation surrogate, but CE plus hidden is lower than the matched FP32 CE control in all three seeds. It beats shuffled targets in only two seeds. We therefore do not establish additional task benefit from correct hidden-response pairing in this setting.

## 2. Router 的真正比较对象

| 方法 | Native 主指标 | Mean accuracy | Worst-task accuracy |
|---|---:|---:|---:|
| Task-ID / own expert | 0.7695 ± 0.0114 | 0.8252 ± 0.0083 | 0.7487 ± 0.0137 |
| Source router | 0.7683 ± 0.0119 | 0.8232 ± 0.0083 | 0.7448 ± 0.0137 |
| Expert router | 0.6559 ± 0.0098 | 0.7559 ± 0.0051 | 0.6497 ± 0.0239 |
| Random | 0.5787 ± 0.0298 | 0.6875 ± 0.0220 | 0.5781 ± 0.0295 |
| Oracle（使用 final GT） | 0.8974 ± 0.0028 | 0.9255 ± 0.0025 | 0.8685 ± 0.0090 |

Expert router 相对 task-ID 的 mean accuracy 变化 **-6.934 个百分点**；oracle 相对 task-ID 约 **+10.026 个百分点**。Oracle说明bank内存在互补预测，不说明base feature能可靠预测何时换expert。我们已知task head，因此直接使用本任务adapter是必须保留的可部署参照。

| 路由 | 帮助了多少输入 | 损害了多少输入 | 净 accuracy 变化 |
|---|---:|---:|---:|
| Source router | 0.098% | 0.293% | -0.195 pp |
| Expert router | 2.637% | 9.570% | -6.934 pp |
| Oracle（使用 final GT） | 10.026% | 0.000% | +10.026 pp |

**英文口答：** The oracle shows complementary predictions, but our learned router does not capture that headroom. It harms more own-expert decisions than it repairs. Source recognition and useful expert selection are different targets, and the known-task baseline is both stronger and cheaper here.

## 3. 成本、数据和统计口径

历史A100实测 individual约23.5ms、GPU-cache router约49.0ms、CPU-delta router约98.6ms（batch1、128tokens）。新L4 router质量没有与L4部署耗时联合测量，不能将其与旧A100毫秒数拼成实测Pareto优势。共享base约1.84GiB、四adapter文件约8.33MiB，97参数仅指新增拟合标量。

误差条是三训练seed的 sample SD，不是样本置信区间，不作显著性宣称。Native均值含CoLA MCC、MRPC F1等，不能读成统一accuracy或完整官方GLUE。Final是原train内部五分区之一，exact tuple去重不解决全部近重复/预训练污染；旧IMDb/PAWS失败检查另列。

## 4. 恢复与复现

11月9日下载中断后，只恢复到18条完整ZIP记录；[原始恢复证据](../artifacts/recovered-partial-2026-11-09/README.md)保留CRC、原manifest SHA与缺失清单。本次从25MB备份恢复131文件/12teacher，完成整轮扩展，并用独立脚本从保存的logits、labels和routes重算所有指标、帮助/损害比例和oracle上界。与恢复CSV对比，最大指标绝对差 **0**。这项重复检查不增加统计样本数。

本轮完整证据： [summary](../artifacts/activation-strengthening-2026-11-11/summary.json)、[配对CSV](../artifacts/activation-strengthening-2026-11-11/paired.csv)、[逐任务CSV](../artifacts/activation-strengthening-2026-11-11/per-task.csv)、[manifest](../artifacts/activation-strengthening-2026-11-11/manifest.json)。每个seed的selection文件保留step、gain、weights；预测npz只含派生logits/labels/routes，不含原文数据或模型权重。

## 5. 如何回答“这算完成原proposal吗”

研究实现、实验、控制和分析已形成完整闭环；原始“预期优于基线”是待检验假设。我们给出有信息量的边界：学习少量系数有用；功能目标未显示超出监督CE的独特帮助；oracle互补性不等于可学习路由优势。更多模型、rank、adapted-layer组合与全新OOD仍属未覆盖扩展，不能概括为已经验证。

**30秒英文总结：** We implemented a controlled study of activation-aware adapter merging. Learned static coefficients recover part of the averaging loss, but output KL does not exceed CE on average. Direct hidden matching also fails to add task benefit, and our expert router remains below task-ID selection despite oracle headroom. The contribution is a reproducible implementation and evidence separating functional proxies, capacity, and useful expert choice.
