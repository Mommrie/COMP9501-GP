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

## 第6页：更大评价区上的主结果（65秒）

**主图：** [官方静态方法比较](../artifacts/robustness-v2-2026-11-11/official-static.png)。标注872/408/277/1023条，known task head，三seed mean±sample SD；任务/官方split历史探索过。

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

> On the larger cleaned validation split, CE reaches 0.7050 native score versus 0.6007 for our TIES variant. Selected aligned and shuffled grids reach 0.7106 and 0.7107, while individual remains at 0.7618. Learning coefficients helps this pipeline, but the matched shuffle result prevents a unique activation claim.

**备份：** 原v3内部CE 0.7132/KL 0.6987与固定基线完整表；两个split不能直接作单因素归因。

## 第7页：λ敏感性与匹配控制（80秒）

**主图：** [λ曲线](../artifacts/robustness-v2-2026-11-11/lambda-sensitivity.png)。同时展示selection与已探索internal final，λ=0即CE。讲清“固定λ曲线”和“每seed按selection选λ的pipeline”是不同汇总。

> Unit hidden weight is too strong in our initial setting. We predeclare smaller weights and a matched shuffled grid. Both grids choose 0.01 only in seed 42 and return to CE in the other seeds. The aligned pipeline gains about 0.0055 native score on cleaned validation, but the shuffled grid is slightly higher. This does not isolate a benefit from correct response pairing.

**追问准备：** 相对单CE增加selection候选；三seed不作显著性宣称；shuffling也打乱class对应，不是只去掉细粒度语义。

## 第8页：保守gate和真实回退（75秒）

**主图：** [官方路由比较](../artifacts/robustness-v2-2026-11-11/official-routing.png) 或 [帮助/误伤](../artifacts/robustness-v2-2026-11-11/official-routing-impact.png)。主讲优先五行：task-ID、expert、confidence、benefit、GT oracle；完整表留备份。

> We add a confidence gate and a direct correctness-gain regressor, each with an always-own fallback. Selection requires native score and accuracy floors; it is not a final guarantee. Both families select task-ID in all three seeds. Some Ridge candidates meet the floors but tie the reference, so the switching and baseline tie rules apply. The result rejects unnecessary routing rather than establishing a new routing advantage.

**实际数值：** 官方macro accuracy：task-ID及两个回退gate82.253%，expert75.647%，oracle91.497%。Oracle使用final GT；缓存四expert离线结果不是部署耗时。两个gate相同柱高来自always-own，不能称新方法获胜。

## 第9页：部署成本与真实存储参照（50秒）

**图：** [A100成本图](../artifacts/part2-followup-2026-11-09/latency.png)。注：旧seed42、synthetic fixed inputs、FP32、batch1/128tokens。

**数值：** individual约23.5ms；GPU-cache router49.0ms；CPU-delta router98.6ms。共享base约1.84GiB，四adapter文件约8.33MiB。

> Static materialization has approximately the same single-forward cost as individual. Our router adds base-feature extraction and one forward per expert group in the batch. Caching reduces state-transfer overhead, but it uses more memory. The original adapter bank already shares one base, so we do not claim savings against four duplicated backbones.

## 第10页：相似度与OOD失败条件（40秒）

**放什么：** Δactivation-pair loss三seedρ：−0.6571/−0.4857/+0.0857；IMDb/PAWS旧检查不是新v3数据。优先简表，不堆全部散点。

> Our similarity diagnostic is not stable across training seeds. Earlier transfer checks also do not establish robustness: IMDb has heavy truncation, and PAWS is below its majority-class reference even for individual teachers. We retain these failures and keep their older protocol separate from the new experiment.

## 第11页：贡献与局限（40秒）

**三项贡献：** 真实实现及可复现协议；把功能信号/容量/来源识别拆开对照；实测质量与成本权衡。

**三项限制：** 一小模型四binary tasks；三seed同一划分且历史探索；已知task head、未建立OOD/隐藏信号独特优势（按最终结果更新最后一句）。

> We contribute a controlled implementation and evidence about when functional proxies are useful or misleading in this setting. Layer-wise merging and distillation have prior work. The limited model, tasks, training seeds and known-head protocol restrict claims about novelty and generalization.

## 第12页：结论（35秒）

> Our study shows that learning a small static mixture can recover part of the averaging loss. The matched-capacity comparison is essential: teacher-output KL does not outperform CE on average. The direct hidden and expert-routing controls further test the proposal's mechanism. We report the resulting evidence and limitations, together with a reproducible pipeline, rather than equating a high proxy score with a successful merge.

最后两句在新增实验核验后，用最终报告给出的明确结果替换概括表述。不要保留未执行实验的预计结论。

第5页数据图另加锁定selection后的官方validation支路：2600原例→去16个train完整tuple重叠及4个区内重复→2580例。它历史上被探索过，不标新盲测。

## 备份页与代码讲解接口

1. 完整八方法/逐任务/全部三seed差值表，来源 [CSV与报告](../PART2-RESULTS-V3-2026-11-09.zh-CN.md)。
2. 新activation objective公式、shuffled permutation、精度/参数/GT权限表。
3. oracle准确率上界、帮助/损害比例、router选择C和实际信息预算。
4. 核心实现链：`common.selected_raw` → `train_lora` → `Coefficients/MixedLinear` → `DeltaEvaluator` → `activation_extension`。
5. [26组中英Q&A](../PART2-IMPLEMENTATION-QA.zh-CN.md)，新增实验的精确结果口答以最终阅读指南为准。

需要组员提供/确认的行政项：最终成员及讲述分工、Group13展示场次、是否有正式报告格式、提交/代码访问安排。不要填猜测的人名、时间或提交回执。

## 路由图表的指标提醒

`official-routing.png` 左图是 native task score：task-ID/gates 0.7618，expert 0.6571，oracle 0.8819。讲稿中的82.253%、75.647%、91.497%是另一个指标 task-macro accuracy，不能直接用作该图柱值。帮助/损害图也按任务宏平均，switch rate按样本加权。
