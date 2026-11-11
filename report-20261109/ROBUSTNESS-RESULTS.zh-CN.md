# 第二轮强化结果：用于最终report与presentation的最新解释

本轮完整跑完三seed、12次新增系数拟合、两类保守gate，以及2580条清理后官方validation复评；28项代码测试、全部保存预测的独立指标审计通过。[英文报告](report.pdf)已扩充方法、对照与讨论；[冻结设计](ROBUSTNESS-PROTOCOL.zh-CN.md)解释预算和数据权限。

## 1. 最适合汇报的研究主线

我们把原proposal分成三个问题，而非用一个高proxy score证明整个想法。学习少量静态系数能恢复平均合并的部分损失；但teacher-output、hidden配对和expert选择分别需要机制控制。第一轮λ=1明显伤害CE，本轮检验较小λ与保守回退，并在更大的单独split复评。结论有正面证据，也有明确边界。

在清理后官方validation，CE native为 **0.7050 ± 0.0220**，TIES为 **0.6007 ± 0.0095**，Average为 **0.5738 ± 0.0178**。选择后的aligned/shuffled目标为 **0.7106 ± 0.0207 / 0.7107 ± 0.0208**，individual仍为 **0.7618 ± 0.0153**。因此支持当前预算下学习系数改善固定基线，尚未建立正确hidden配对的独特帮助或超过individual的收益。

## 2. 完整官方validation比较

| 方法 | Native均值±sample SD | Mean accuracy | Worst accuracy |
|---|---:|---:|---:|
| Individual / task-ID | 0.7618 ± 0.0153 | 0.8225 ± 0.0164 | 0.7353 ± 0.0623 |
| Average | 0.5738 ± 0.0178 | 0.7111 ± 0.0033 | 0.5391 ± 0.0205 |
| TA | 0.5874 ± 0.0170 | 0.7025 ± 0.0063 | 0.5571 ± 0.0149 |
| TIES | 0.6007 ± 0.0095 | 0.7024 ± 0.0066 | 0.5223 ± 0.0290 |
| DARE+TA | 0.5952 ± 0.0277 | 0.6971 ± 0.0120 | 0.5387 ± 0.0578 |
| CE FP32 | 0.7050 ± 0.0220 | 0.7803 ± 0.0174 | 0.6750 ± 0.0719 |
| Selected aligned H | 0.7106 ± 0.0207 | 0.7807 ± 0.0179 | 0.6703 ± 0.0662 |
| Selected shuffled H | 0.7107 ± 0.0208 | 0.7806 ± 0.0177 | 0.6703 ± 0.0662 |
| Source router | 0.7552 ± 0.0137 | 0.8178 ± 0.0153 | 0.7316 ± 0.0646 |
| Expert router | 0.6571 ± 0.0219 | 0.7565 ± 0.0167 | 0.6462 ± 0.0407 |
| Confidence gate | 0.7618 ± 0.0153 | 0.8225 ± 0.0164 | 0.7353 ± 0.0623 |
| Benefit gate | 0.7618 ± 0.0153 | 0.8225 ± 0.0164 | 0.7353 ± 0.0623 |
| Oracle（final GT） | 0.8819 ± 0.0165 | 0.9150 ± 0.0146 | 0.8437 ± 0.0503 |

Native是SST-2/RTE accuracy、MRPC(acc+F1)/2、CoLA MCC的四任务macro均值；不是统一accuracy或完整GLUE score。不同任务的样本量不同，mean accuracy仍是任务macro均值；switch fraction则按全部输入计数。

## 3. 权重网格到底说明什么

λ∈{0,.01,.1,1}，真实/shuffled目标共享网格、同97参数、同FP32与120步。0/1端点复用已完成结果，新增中间权重。每λ按selection选step，再按selection选λ；相同分数偏向较小λ。

| Seed | Aligned λ | Shuffled λ | Official H−CE | Official shuffle−CE | Official H−shuffle |
|---|---:|---:|---:|---:|---:|
| 42 | 0.01 | 0.01 | +0.016626 | +0.016935 | -0.000309 |
| 43 | 0.0 | 0.0 | +0.000000 | +0.000000 | +0.000000 |
| 44 | 0.0 | 0.0 | +0.000000 | +0.000000 | +0.000000 |

Aligned-grid − CE均值 **+0.005542**；shuffled-grid − CE为 **+0.005645**。只有seed42使用λ=.01，其余回到CE。Shuffled稍高于real，不能把约0.0055的native提升称正确activation配对的独特收益。较小λ缓解λ=1的破坏，但并未验证原机制。

网格相对CE单轨迹拥有更多selection机会：最多4λ×5step候选（其中有重复）。两种目标网格预算匹配，CE的调参预算更少。三个seed误差条是sample SD，不是置信区间，也不作显著性宣称。重复拟合不是新增teacher seed。

**英文口答：** A smaller response weight recovers some of the unit-weight loss, but the aligned grid does not beat its matched shuffled grid. Both select a small weight in only one seed and revert to CE in two. The selected-pipeline gain therefore does not establish a unique benefit from correct response pairing.

## 4. Gate回退不是“新router赢了”

两类gate全部选择always-own（benefit为3/3；confidence也为3/3），最终切换率0，质量等于task-ID。Confidence候选均未通过声明的selection质量约束；部分强正则Ridge候选通过约束但与参考持平，少切换或baseline优先的平局规则决定选择。不能说所有Ridge候选都危险，也不能把不切换称新路由准确率提升。

原expert router在官方区修复的macro输入比例为 2.651%，误伤 9.258%，mean accuracy为75.647%；task-ID为82.253%。Oracle达到91.497%，但用了final GT。专家互补不代表我们选用的base features和有限fit数据足以预测这种互补。

Confidence预测的是expert身份的置信度；benefit Ridge直接拟合expert correctness−own correctness（−1/0/+1）。后者更贴近目标，但小fit集、高维线性特征、正则和校准仍限制它。当前失败不证明所有router不可能。

**英文口答：** The conservative selectors choose the available task-ID reference in all three seeds. Some benefit candidates meet the selection floors but tie the reference, so the switching and baseline tie rules apply. The result is rejection of unnecessary routing, not a new routing accuracy advantage; a selection floor would not guarantee safety if a finite gate were selected.

## 5. 更大评价区与数据过滤

| Task | 原validation | 保留 | 与全部原train完整tuple重复 | validation内重复 |
|---|---:|---:|---:|---:|
| sst2 | 872 | 872 | 0 | 0 |
| mrpc | 408 | 408 | 0 | 0 |
| rte | 277 | 277 | 0 | 0 |
| cola | 1043 | 1023 | 16 | 4 |

标签不参与过滤；NFKC/casefold/空白归一化的完整字段tuple用于对照四任务所有原train，再在每任务validation内去重。CoLA去掉16条train内容重叠和4条区内重复，总数由2600变成2580。近重复、句对内单句重用和预训练污染仍未完全排除。

这是更大、单独split的检查；GLUE官方validation在旧研究中已看过，不称全新盲测或新OOD证明。λ/gate只按原selection选，先写lock，再读取缓存final数组与载入官方validation；内部final成绩此前已存在并启发研究，完整历史不会被重命名抹去。

## 6. 逐任务备份表

| Task（主指标） | Individual | CE FP32 | Selected aligned H | TIES |
|---|---:|---:|---:|---:|
| SST-2 acc | 0.9182 ± 0.0063 | 0.9041 ± 0.0156 | 0.9037 ± 0.0152 | 0.8899 ± 0.0248 |
| MRPC (acc+F1)/2 | 0.8554 ± 0.0025 | 0.8075 ± 0.0169 | 0.8078 ± 0.0171 | 0.6823 ± 0.0889 |
| RTE acc | 0.7353 ± 0.0623 | 0.6775 ± 0.0752 | 0.6703 ± 0.0662 | 0.5223 ± 0.0290 |
| CoLA MCC | 0.5383 ± 0.0153 | 0.4310 ± 0.1125 | 0.4605 ± 0.0665 | 0.3083 ± 0.0538 |

完整accuracy/F1/MCC、全部seed、λ、阈值和帮助/误伤见 [原始CSV与JSON](../artifacts/robustness-v2-2026-11-11/README.md)。四张新图可直接给PPT制作者：λ曲线、官方静态比较、官方router比较、实际帮助/误伤。主讲只取关键图和少量行，完整表留备份页。

## 7. 教授追问：十组精确口答

1. **有提升，为什么不说activation成功？** “The matched shuffled grid is slightly higher, two seeds choose CE, and the grid has more selection candidates. The evidence supports the selected pipeline, not a unique alignment mechanism.”
2. **为什么允许λ=0和always-own？** “They are valid available references. Including them lets calibration reject unnecessary complexity. It prevents forced adoption on selection, but does not guarantee final performance for a selected finite rule.”
3. **官方validation为何不能称盲测？** “It is separate from the v3 fit roles, but we explored the same benchmark historically. We disclose that history and use it as a larger split check rather than a pristine test.”
4. **0.7106是不是71.06% accuracy？** “No. It is a macro native metric containing CoLA MCC and MRPC F1. We separately report task-macro accuracy and each task's metrics.”
5. **为何Ridge有eligible候选仍选own？** “Some meet the quality floors but tie native and accuracy; fewer switches and baseline tie preference retain the reference. Eligibility is not the same as a useful improvement.”
6. **新12次拟合是否增加seed？** “No. They reuse the same twelve teachers and three training seeds. The grid explores coefficient objectives; it does not add independent teacher replications.”
7. **怎么确保没按final挑参数？** “The combiner uses selection fields, writes its lock, and only then reads final prediction arrays and loads official validation. Earlier final inspection still makes the follow-up exploratory.”
8. **回退是不是性能或存储优化？** “Always-own can bypass feature extraction, but that is the existing baseline. We do not combine L4 quality with old A100 milliseconds or compare storage against four duplicated backbones.”
9. **Shuffled只破坏细粒度input pairing吗？** “It is a within-task derangement that also breaks class-information correspondence. It preserves the marginal target set; a class-conditional shuffle would be a separate, untested control.”
10. **自己的贡献在哪里？** “We implement the constrained LoRA mixture and controlled hypotheses, explicit data permissions, matched shuffle and CE controls, conservative selection, larger-split checks and prediction audits. Layer-wise merging and distillation have prior work; CSA is conceptual motivation rather than an implemented LLM projection.”

## 8. 一分钟英文总结

> We investigate whether functional signals help merge task-specific LoRA updates. Learned static mixtures improve our fixed weight-space baselines on both the internal protocol and a larger, content-filtered official validation split. However, output KL does not beat CE on average, and a smaller hidden-loss grid is matched by its shuffled control. Two seeds choose no hidden penalty. Our conservative routers also select the known-task reference rather than useful new switches. The project therefore contributes a reproducible implementation and evidence separating capacity, calibration and functional proxies, with explicit limits on novelty, supervision and generalization.

## 选文与Part1的直接桥接（新增）

新增两页PPT安排、完整三seed结果、代码与论文公式对应、英文口述及12组追问见 [CSA桥接材料](CSA-BRIDGE-RESULTS.zh-CN.md)。这部分直接应用CSA压缩expert响应，原合并主线结论保持原样；response compression不等于weight compression或推理加速。
