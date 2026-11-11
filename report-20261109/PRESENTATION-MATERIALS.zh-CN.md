# 给PPT制作者：Part 2逐页素材、讲述与分工接口

这份文档提供内容和讲解，不代替组员的正式幻灯片。以原proposal的三个问题为主线：相似度能否预测干扰、功能信号能否改善静态合并、动态路由能否选到更有用expert。不要沿用Part1的CSA/PCA实验结果作为Part2证据。

建议12页主讲约10分20秒，另留40秒切换缓冲及4分钟QA，合计15分钟；这是演练方案，不是已确认的课程分配。若组内另有安排，按实际场次调整。

## 第1页：题目与一句话问题（25秒）

**Title:** Activation-Aware Merging of LoRA Adapters for Multi-Task Learning

**放什么：** 一个共享模型、四个任务adapter，箭头指向单模型或router。副标题可用“A controlled study of functional signals and interference”。

**英文讲述：**

> Our project asks whether functional information can help combine task-specific LoRA adapters. We investigate three separate questions: predicting merge interference, learning a static merge, and choosing experts per input.

**不要说：** 提出了全新SOTA；Part2直接实现了CSA投影。

## 第2页：从原proposal到可检验问题（40秒）

**三行：** Similar activations → safe merge? Functional targets → useful coefficients? Dynamic routing → better expert choices?

**依据：** [proposal原文](../PROPOSAL.md) Methodology/Expected Outcomes；把“预期更好”标为假设，不作结论。

> Our initial hypothesis was that similar adapter responses indicate compatible updates, while conflicting cases may benefit from gating. We turn this into controlled comparisons instead of assuming that a high similarity score or a successful implementation validates the whole idea.

**追问准备：** 相关工作已包括AdaMerging/RegMean/KnOTS；项目贡献是研究设计、具体实现及对照分析，见 [相关工作定位](../PART2-RELATED-WORK.zh-CN.md)。

## 第3页：模型、adapter与任务头（50秒）

**放什么：** Qwen2.5-0.5B；SST-2/MRPC/RTE/CoLA；LoRA r=8，q/v投影；每任务自己的冻结head。

**公式：** `ΔW_t=(α/r)B_t A_t`。强调合并BA更新，不分别平均A/B。四个rank8之和不必仍为rank8。

> Each task trains a low-rank update and its own classification head on the same pretrained backbone. During merging, all those parameters are frozen. We merge actual weight updates, and the known task ID selects the appropriate head. This is a task-aware setting, even though all four tasks are binary.

**追问准备：** 为什么不统一head？二分类维度相同不等于标签语义相同；未知task需新输出协议。

## 第4页：静态方法与容量对照（65秒）

**放什么：** `W_m,l=W_0,l+g Σ_t softmax(a_l)_t ΔW_t,l`；24×4+1=97标量，global为5。图示冻结部分与训练系数的颜色不同。

> We optimize a small mixture for each layer and one shared gain. The layer-wise version stores ninety-seven trainable scalars; the backbone, adapter factors, and task heads do not receive updates. The final merge is materialized, so serving uses one backbone forward. We distinguish this fitting parameter count from the full model footprint.

**追问准备：** softmax有平移冗余；97是张量参数数，不是97个独立自由度。正权重/gain范围限制了可表达合并。

## 第5页：信息权限与复现（55秒）

**放什么：** 五盒流程：Train → Teacher select → Merge fit → Merge select → Final；每条训练/选择箭头只接到它有权限的模块。标注“原train内部划分”。

**数据：** train=4096/2391/1571/4096；各任务后四区128/128/128/256；seeds42/43/44。

> We retrain twelve teachers with separate data for teacher checkpoint selection, merge fitting, merge selection, and final evaluation. Complete text tuples are grouped before partitioning. Revisions, sample IDs and checksums are recorded. These are internally isolated partitions of previously studied GLUE tasks, not a new external blind test.

**追问准备：** 旧calibration重切不能恢复teacher选择独立性；新扩展看到过v3 final，因此诚实称探索性。

## 第6页：主结果与真正的对照（65秒）

**图：** [三seed方法比较](../artifacts/independent-v3-2026-11-09/main-comparison.png)。空间不足时主表保留Individual、Average、TIES、Layer CE、Layer KL，完整八方法放备份页。

| 方法 | 主指标均值±sample SD |
|---|---:|
| Individual | 0.7695±0.0114 |
| Average | 0.5578±0.0246 |
| TIES（本实现） | 0.6146±0.0240 |
| Layer CE | 0.7132±0.0225 |
| Layer KL | 0.6987±0.0080 |

> Learned coefficients improve our fixed weight-space grid, but the same-capacity CE control exceeds teacher-output KL on average. KL is higher in only one of three paired training seeds. This means the improvement over averaging cannot be uniquely attributed to teacher-output supervision, and both learned merges remain below individual models.

**脚注：** native mean，不是完整GLUE score；CE还使用fit GT；不同baseline总监督/计算预算不同。

## 第7页：直接activation检验（80秒）

**放什么：** 新扩展五profile：FP32 CE、hidden-only、CE+hidden、CE+shuffled hidden、global hidden。新增结果来自11月11日L4完整复核，不从旧KL表推断。

**关键图示：** 同一输入的teacher delta连到student；对照中换成同任务另一条输入的delta。显示“same targets, broken pairing”。

> Output KL does not directly test hidden activation information. We therefore add a normalized hidden-response objective and a fixed within-task shuffle control. CE plus real and shuffled responses share precision, capacity, and update schedule. The comparison tests whether input correspondence contributes beyond the response distribution. The global variant also probes the role of layer-wise flexibility.

| 方法 | Native 主指标 |
|---|---:|
| CE FP32（97参数） | 0.7130 ± 0.0211 |
| Hidden-only（97） | 0.5901 ± 0.0259 |
| CE + hidden（97） | 0.6397 ± 0.0281 |
| CE + shuffled（97） | 0.6195 ± 0.0056 |
| Hidden-only（5） | 0.5841 ± 0.0297 |

> CE plus aligned hidden is lower than FP32 CE by 0.0733 native score on average, with no positive seed. Its advantage over shuffled targets is inconsistent and does not establish benefit beyond CE.

**结论呈现规则：** 先讲真实vs同精度CE，再讲真实vsshuffled；若无提升，直接说明没有支持额外收益。MSE下降不能代替分类质量改善。新数据和图表必须标“exploratory extension”。

## 第8页：Router究竟在预测什么（75秒）

**放什么：** source ID target与minimum-NLL expert target两个不同标签；比较task-ID、source router、expert router、random、GT oracle。

> Recognizing an input's source is different from selecting the expert that helps its prediction. We compare a source classifier with a performance-oriented router trained on the best expert under fit labels. Because the task head is known, direct task-ID selection is the required baseline. A final-label oracle shows the bank's accuracy ceiling, not a deployable result.

| 方法 | Mean accuracy | Worst-task accuracy |
|---|---:|---:|
| Task-ID / own expert | 0.8252 ± 0.0083 | 0.7487 ± 0.0137 |
| Source router | 0.8232 ± 0.0083 | 0.7448 ± 0.0137 |
| Expert router | 0.7559 ± 0.0051 | 0.6497 ± 0.0239 |
| Random | 0.6875 ± 0.0220 | 0.5781 ± 0.0295 |
| Oracle（使用 final GT） | 0.9255 ± 0.0025 | 0.8685 ± 0.0090 |

> The expert router is 6.93 accuracy points below task-ID selection, while the final-label oracle has 10.03 points of headroom. This is a failure to learn useful choices, despite complementary experts.

**Oracle只作分析上界。** 图的oracle用灰色/虚线，显著标注“uses final GT”。同时讲helps/harm比例，而不是只看来源accuracy。

**追问准备：** expert router需要fit GT和离线四expert前向；本轮缓存评估不是部署耗时。

## 第9页：部署成本与真实存储参照（50秒）

**图：** [A100成本图](../artifacts/part2-followup-2026-11-09/latency.png)。注：旧seed42、synthetic fixed inputs、FP32、batch1/128tokens。

**数值：** individual约23.5ms；GPU-cache router49.0ms；CPU-delta router98.6ms。共享base约1.84GiB，四adapter文件约8.33MiB。

> Static materialization has approximately the same single-forward cost as individual. Our router adds base-feature extraction and one forward per expert group in the batch. Caching reduces state-transfer overhead, but it uses more memory. The original adapter bank already shares one base, so we do not claim savings against four duplicated backbones.

## 第10页：相似度与OOD失败条件（40秒）

**放什么：** Δactivation-pair loss三seedρ：−0.6571/−0.4857/+0.0857；IMDb/PAWS旧检查不是新v3数据。优先简表，不堆全部散点。

> Our similarity diagnostic is not stable across training seeds. Earlier transfer checks also do not establish robustness: IMDb has heavy truncation, and PAWS is below its majority-class reference even for individual teachers. We retain these failures and keep their older protocol separate from the new experiment.

## 第11页：贡献与局限（40秒）

**三项贡献：** 真实实现及可复现协议；把功能信号/容量/来源识别拆开对照；实测质量与成本权衡。

**三项限制：** 一小模型四binary tasks；三seed同一划分且历史探索；已知task head、未建立OOD/隐藏信号独特优势。

> We contribute a controlled implementation and evidence about when functional proxies are useful or misleading in this setting. Layer-wise merging and distillation have prior work. The limited model, tasks, training seeds and known-head protocol restrict claims about novelty and generalization.

## 第12页：结论（35秒）

> Our study shows that learning a small static mixture can recover part of the averaging loss. The matched-capacity comparison is essential: teacher-output KL does not outperform CE on average. Direct hidden matching is 0.0733 native score below FP32 CE, and expert routing is 6.93 accuracy points below task-ID selection. These results delimit the original hypothesis and show why functional proxies and oracle complementarity need explicit task-quality controls.

以上结论已填入完整复核数值；逐任务差异、帮助/损害比例和30秒口答见 [最终结果精讲](ACTUAL-RESULTS.zh-CN.md)。

## 备份页与代码讲解接口

1. 完整八方法/逐任务/全部三seed差值表，来源 [CSV与报告](../PART2-RESULTS-V3-2026-11-09.zh-CN.md)。
2. 新activation objective公式、shuffled permutation、精度/参数/GT权限表。
3. oracle准确率上界、帮助/损害比例、router选择C和实际信息预算。
4. 核心实现链：`common.selected_raw` → `train_lora` → `Coefficients/MixedLinear` → `DeltaEvaluator` → `activation_extension`。
5. [26组中英Q&A](../PART2-IMPLEMENTATION-QA.zh-CN.md)，新增实验的精确结果口答以最终阅读指南为准。

需要组员提供/确认的行政项：最终成员及讲述分工、Group13展示场次、是否有正式报告格式、提交/代码访问安排。不要填猜测的人名、时间或提交回执。
