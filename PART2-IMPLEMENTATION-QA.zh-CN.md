# Part 2 实现讲解与教授问答

适用 Group 13 的 Qwen2.5-0.5B / GLUE / LoRA 合并项目。与 Part 1 CSA 的问题不要混用。数值结论以本轮结果报告和原始 JSON 为准；以下口答解释实际代码和实验权限。

本轮速记：三 seed 主指标均值 individual **0.7695**、TIES **0.6146**、Layer CE **0.7132**、Layer KL **0.6987**。学习系数改善了固定权重基线，但 KL−CE 平均为 **−0.014468**，仅 1/3 seed 较高，未建立 teacher-output 的独特优势。全部误差与逐任务结果见 [新协议完整结果](PART2-RESULTS-V3-2026-11-09.zh-CN.md)。CE/KL 与权重基线的 fit 数据/监督权限不同，不把管线比较说成完全相同的信息预算。

教授问本轮结论时可答：

> Learned static coefficients improved our fixed weight-space baselines, but teacher-logit KL did not outperform the matched-capacity CE control on average. Both remain below the task-aware individual models. This supports calibration-based merging in our setting, but does not establish a unique benefit from the proposed functional signal.

## 60 秒英文流程

> We start from the same frozen Qwen backbone and train a separate LoRA adapter and classification head for each task. In the new protocol, we separate teacher training, teacher checkpoint selection, coefficient fitting, coefficient selection, and final evaluation using content-grouped partitions. We then freeze the backbone, adapters, and heads, and learn only layer-wise merging coefficients. We compare supervised cross-entropy with teacher-logit matching at the same parameter count and update budget. After selecting the coefficients, we materialize one merged backbone and evaluate it with the appropriate frozen task head. We report per-task quality, worst-task accuracy, training-seed variation, and deployment costs, including negative results and protocol limitations.

## 三分钟代码讲解顺序

1. [independent_protocol.py](code/independent_protocol.py)：从源 train 按完整字段文本 tuple 分组，生成五区 manifest。解释为何只把旧 calibration 切半不够：旧 teacher 选 checkpoint 已看过它。
2. [common.py](code/common.py) 的 `selected_raw`：检查 manifest SHA 与 dataset fingerprint，按 IDs 读取明确 role；`encode_example` 保留句对角色、最长 128 tokens。`validation` 在 v3 是内部最终区名字，实际来自原 train。
3. [train_lora.py](code/train_lora.py)：每任务更新 q/v 的 LoRA 和独立 score head；共享 backbone 冻结，teacher 只在自己的 selection 区选 epoch。三个 seed 分别从头训练。
4. [coefficient_merge/core.py](code/coefficient_merge/core.py)：`Coefficients` 保存 24×4 logits 与 gain；`MixedLinear` 用低秩乘法计算混合后的响应，冻结 A/B 与 base。
5. [coefficient_merge/run.py](code/coefficient_merge/run.py)：比较 CE 与 T=2 的 teacher KL，使用同样抽样顺序和 120 steps；只在 coefficient_selection 挑 step，保存选择文件后才读 final。
6. [merge_study/engine.py](code/merge_study/engine.py)：用 immutable base snapshot 加入 ΔW，避免重复切换时更新累计；分类头按 task ID 替换，不参与平均。
7. [independent_baselines.py](code/independent_baselines.py) 与 [summarize_independent_study.py](code/summarize_independent_study.py)：固定权重基线、三 seed 配对差值、数据权限复核、逐任务 CSV 与图表。

## 1. 你们究竟提出了什么问题？

> We ask whether functional information helps merge task-specific LoRA updates under a small, fixed merging capacity. We separate similarity diagnostics, coefficient learning, and routing because they test different claims.

原 proposal 比较宽；现在静态主线比较“teacher 输出是否提供超出相同容量 CE 的帮助”。相似度相关和 router 不是同一实验，不能把其中一个成功当成另外两个已成立。

## 2. 你们哪里不是照抄论文？

> Our contribution is an implemented and controlled empirical study, rather than a claim that layer-wise coefficients or distillation are new. We test the original hypothesis, add matched-capacity controls, isolate data permissions, and measure practical routing costs.

主动承认 AdaMerging、RegMean、KnOTS 等先例，见 [相关工作](PART2-RELATED-WORK.zh-CN.md)。课程研究可以贡献清楚的问题、可复现实现与失败条件；不能为了显得新颖，把已有概念改名字。

## 3. Part 1 CSA 是否实际用于 Part 2？

> CSA motivated our interest in structured representations and explicit resource budgets. Our current Part 2 implementation does not apply CSA projections to LoRA weights; it learns coefficients over materialized task updates.

不要说“我们把 CSA 用进了大模型”。当前没有 mode basis、CSA 交替特征分解或 CSA 重建目标；代码中的层级系数是另一个参数化。

## 4. LoRA 为什么只训练很少参数？

> A LoRA update factors a weight change as B times A, scaled by alpha over rank. We train these low-rank factors and the task head while keeping the pretrained backbone fixed.

对 W∈R^(out×in)，A∈R^(r×in)、B∈R^(out×r)，可训练因子数为 r(in+out)，再加任务头。实际总数以模型打印和参数张量为准，不拿 97 个合并标量冒充 teacher 训练参数。

## 5. 为什么不能直接平均 A 和 B？

> Averaging A and B separately introduces cross-adapter products. We first form each actual weight update, then merge those updates, which matches the intended model arithmetic.

两任务时 `(B1+B2)(A1+A2)/4` 包含 B1A2、B2A1；目标平均是 `(B1A1+B2A2)/2`。已有 [数学检查](code/test_math.py) 验证物化更新与合并操作。

## 6. 合并后还是 rank 8 吗？

> Not necessarily. A sum of four rank-eight updates can have rank up to thirty-two for a layer. We evaluate a materialized merged backbone; we do not claim the result is a single rank-eight adapter.

实际 rank 还受矩阵维度和更新之间相关性限制。若另做 rank-8 压缩，要新增 SVD 近似、压缩预算和重评；当前不能默认为无损。

## 7. 为什么是 97 个参数？

> We store four mixing logits for each of twenty-four layers, plus one shared gain. That is ninety-seven trainable scalars; the backbone, LoRA factors, and task heads remain frozen during merging.

softmax 每层有公共平移冗余，97 是张量参数数，不是 97 个独立自由度。该数不等于模型存储量。最终推理仍需 backbone 和任务头。

## 8. CE 和 KL 的区别是什么？

> CE fits ground-truth class labels. KL fits the frozen individual teacher's softened output distribution. We keep the coefficient capacity, sample order, optimizer settings, update count, and checkpoint-selection access matched.

KL 使用 T=2 和 T² scaling；teacher 错误也可能被继承。二分类 soft distribution 提供的信息较有限，不能预设 KL 应该胜出。

## 9. KL 是否等于 activation matching？

> Our KL variant matches final classification logits, not all intermediate hidden activations. It is a functional-output control, so its result alone cannot validate a hidden-activation similarity hypothesis.

旧 probe 的 h_adapter−h_base、局部线性响应匹配和最终 logits KL 是三种信号，展示时明确区分。

## 10. KL 不用标签，是无监督方法吗？

> Only the coefficient-update target is teacher-generated. Teacher training uses labels, and both CE and KL checkpoints are selected with labels on the same separate selection partition. We therefore do not call the full pipeline unsupervised.

旧 teacher-KL selection 控制减少一种选择标签使用，但不消除 teacher 训练标签和旧 validation 探索历史；不能据此升级为全流程无监督。

## 11. 为什么要重训 teacher？

> The old teacher checkpoints were chosen using the original calibration pool. Splitting that pool afterwards would not make it independent. We retrain teachers with a dedicated checkpoint-selection partition and reserve separate data for merging and final evaluation.

这是修复信息权限，不能说旧实验完全无效；旧结果仍是探索记录。

## 12. 五分区怎样防止泄漏？

> We assign normalized full-text tuples to roles by a fixed content hash, without looking at labels. We check sample IDs and cross-task content hashes, pin the dataset revision, and lock the partition manifest with a checksum.

完整句对 tuple 去重不等于语义去重。单句局部重复、近重复及预训练数据污染未全部排除。三 seed 共用一个固定划分，不能当三次数据独立重复。

## 13. 这次是不是全新的盲测？

> It is an internally isolated final partition for this new training protocol, but the GLUE datasets and earlier validation results have already informed our research. We do not describe it as a pristine external blind test.

本轮 final 原始来源是 train，代码兼容名叫 validation；展示必须标注 new internal final partition，避免误说“官方完整 validation”。

## 14. 为什么还要 CE 对照？

> A learned merge may improve simply because it has more flexibility or supervised selection. The matched CE control tests whether teacher-output targets provide a benefit beyond that capacity and optimization budget.

如果 KL 与 CE 接近或更差，结论就是未建立额外帮助。若 KL 胜出，也只能在当前任务、模型、容量和权限内报告，不能替代隐藏层相似度或未知任务路由的证据。

## 15. Average / TA / TIES / DARE 怎么实现？

> Average takes the mean of materialized updates. Task Arithmetic scales their sum. Our TIES variant trims each matrix, elects signs, and takes a disjoint mean. DARE applies a fixed random drop-and-rescale mask before Task Arithmetic.

四任务时 average 等价于 scale=0.25 的 TA。TIES 是明确的 per-matrix 变体，DARE mask seed 固定 42；不冒充官方所有实现细节。各方法候选数不同，只有选择样本权限一致，不能说选择算力完全公平。

## 16. 为什么保留每任务 head？

> The tasks have different label meanings, even though all four are binary. We use the known task ID to select a frozen task-specific head and merge only backbone updates. This is a task-aware multi-task evaluation.

共享二分类维度不代表共享语义空间。未知任务推理需要额外定义 head 选择或统一输出；当前没有解决。

## 17. Router 98% 来源准确率为什么不能说明更好？

> Predicting where an input came from is different from choosing the expert that gives the best task prediction. Dataset style and sentence structure can reveal source identity. Moreover, the task ID is already available in our current head protocol.

当前 task-aware 设定本来可直接选择对应 adapter。去标题后来源准确率仍高，只说明标题不是唯一线索；不能当未知任务泛化证据。

## 18. Router 多花多少计算？

> It needs a base-feature pass and then one forward pass for each expert group in the batch. GPU delta caching reduces transfer overhead but consumes additional memory and does not remove the feature pass.

本轮旧 seed42 A100、batch1/128 tokens 的实测：individual 约23.5ms、GPU缓存49.0ms、CPU切换98.6ms；具体测量条件见 [成本报告](PART2-FOLLOWUP-2026-11-09.zh-CN.md)。不要把这些 synthetic timing 当新协议的 task accuracy。

## 19. 你们的存储优势是什么？

> Static materialization simplifies execution to one merged backbone, but our original LoRA bank already shares one base model. We count the base, adapters, heads, coefficients, and any dense GPU caches separately instead of comparing against four duplicated backbones.

当前四个 adapter 文件约8.33MiB，完整 base 参数约1.84GiB；缓存展开后的 dense ΔW 是另一种存储表示。静态合并的主要实测特点是单次 backbone 执行，不能夸成大幅压缩共享-base bank。

## 20. 为什么不只报告平均 accuracy？

> Mean accuracy can hide a task that collapses. We report each task, worst-task accuracy, native task metrics, and the drop relative to the corresponding individual model. CoLA uses MCC as its primary metric, and MRPC uses the mean of accuracy and F1.

不要把异尺度 native metric 的最小值当“最差任务”统一解释。accuracy drop 是相对 individual 的性能损失，不是经过顺序训练定义的 continual-learning forgetting。

## 21. 三个 seed 够证明统计显著吗？

> Three seeds expose some training variability but provide limited statistical evidence. We show all paired CE/KL results and sample standard deviations; we do not claim significance or broad generalization from this sample size.

同任务的六个 pair 共享 adapter，18 个 pair×seed 不是18个独立数据集。sample SD 不是 confidence interval。

## 22. 相似度结果为什么不稳定？

> Similar hidden responses may reflect the shared pretrained backbone rather than compatible task updates. Subtracting the base response makes the signal more specific, but our three-seed pairwise results are not stable enough to establish a reliable merge rule.

这也是为何从 raw activation 转看 Δactivation、ΔW，并保留没有支持假设的结果。不能把事后最好的 seed 当结论。

## 23. 如果 KL 没胜过 CE，项目还有什么价值？

> Then the matched-capacity experiment does not support a unique benefit from teacher-output matching in this setting. We can still report a reproducible implementation, identify misleading proxy metrics, quantify deployment trade-offs, and explain what the evidence rules out.

讲述“假设—受控验证—结论收紧”，不要把负结果改写成性能优势。课程交付是否满足要求还要按实际 rubric 和组内任务核对。

## 24. 为什么没有把所有新模型和 OOD 都跑一遍？

> We prioritized controls that determine whether the central claim is valid. Earlier IMDb and PAWS checks did not establish robust transfer, so we retain them as negative evidence rather than expanding benchmarks without a clear hypothesis.

新 v3 的内部最终区不能替代 OOD；旧 OOD 也不能与新 teacher 的成绩混为一套实验。进一步外推需要独立目标和固定评估协议。

## 25. 如何确认代码没有偷偷训练整个 backbone？

> During coefficient fitting we freeze the backbone, LoRA factors, and task heads, optimize only the coefficient module, verify nonzero coefficient gradients, and assert that frozen parameters receive no gradients. We also check the low-rank forward against materialized dense-update logits.

teacher 训练阶段和 coefficient 阶段是两个优化任务：前者更新 LoRA+head，后者只更新97标量。工程检查证明实现对应计算设计，不保证科学假设成立。

## 26. 教授让你现场指出真正的限制

> We know the task head, study one small backbone and four binary tasks, use only three training seeds on one fixed partition, and have already explored these datasets. Our header edit does not remove all source cues, and our timing is for a specific float32 implementation. These limits restrict our claims about novelty, unknown-task routing, and generalization.

遇到未实现问题直接说 “We have not implemented that setting.” 然后指出要改的具体协议：未知任务需要 head，rank压缩需要重新评价，外部泛化需要新目标，真正 novelty 需要与直接先例逐项比较。
