# 课程报告阅读指南：如何把代码、proposal和结果讲成一个研究故事

## 先掌握这条主线

原proposal的问题是：四个任务各有LoRA adapter，平均合并会不会损伤能力？能否利用功能响应来决定合并，必要时按输入选择expert？我们已经把它实现为一套有对照的研究流程。现在要分别回答“相似度是否有预测价值”“功能目标是否有增量帮助”“路由是否真的选到更有用模型”，不能把它们当成同一个已证实的结论。

目前已经有完整v3实验和成本诊断；本轮新增直接hidden与best-expert实验已在云端L4启动，12个teacher恢复校验和15项开跑检查通过，尚待完整三seed结果。A100未获得分配，所有新增FP32控制使用同一L4，旧A100结果保持分开。不要把执行中的结果读成已经完成的结论。阅读顺序建议：本页 → [实验/主张映射](EXPERIMENT-AND-CLAIM-MAP.zh-CN.md) → [现有v3结果](../PART2-RESULTS-V3-2026-11-09.zh-CN.md) → [英文报告源码](report.tex) → [逐页素材](PRESENTATION-MATERIALS.zh-CN.md)。

## 1. Proposal是研究假设，不是必须兑现的胜率承诺

原文写“expect activation-aware merging to outperform”，这是要检验的预期。项目完成的标准是方法被准确实现、对照能区分机制、数据权限清楚、结果可复现、结论符合证据。不能为了符合预期而挑选最好的seed、忽略CE控制或将旧validation重命名成新盲测。

目前可支持的正面结论是：在固定小模型/四任务/校准预算下，学习少量静态系数能改善我们的固定权重网格。更强的结论“activation独特有效”“路由更便宜”“OOD稳健”还没有成立。负结果并不抹去已经实现的研究，但需要清楚解释它排除了什么说法。

## 2. 代码到底合并什么

每个任务更新`ΔW=(α/r)BA`，这里r=8、α=16，模块是Qwen的q/v projection。合并器用真实ΔW做加权和，不分别平均A与B。原因是分别平均因子会产生B1A2等跨任务乘积，偏离目标更新。

24层分别保存四个任务的softmax logits，再加一个共享gain，共97个可训练标量。它们控制“各任务更新在本层占多大比例”，不是97个隐藏特征，也不是压缩后的完整模型。全局版本所有层共享同一四任务比例，因此只有5个标量。

训练合并系数时，base、LoRA因子和各任务head都冻结。最后把选定ΔW物化进backbone，推理时执行一套网络。合并四个rank8更新后矩阵rank可以上升到32，当前没有再压回rank8。模型权重平均也不等于四个模型预测平均：整条网络是非线性的。

## 3. 为什么一定要CE控制

原先KL变体用teacher最终分类输出作为目标，让合并模型模仿individual teacher。它可以改善Average，但这可能来自更灵活的逐层系数、额外优化和selection，而不是teacher功能信号的独特价值。

所以要问：如果同样97个标量、同样120步、同样输入次序，直接用GT标签的CE是否一样好？v3得到CE主指标0.7132，KL0.6987；KL−CE三seed分别+0.014052、−0.045911、−0.011545，平均−0.014468。不能再把相对Average的提升全部归因于KL。

这不意味着“KL对所有问题无用”。它仅说明本设置未建立KL相对这项控制的平均优势。KL的fit梯度不用GT，但teacher训练和selection仍使用GT，所以整个系统不是无监督。

## 4. 为什么本轮还加hidden与shuffle

KL匹配的是最终logits，原proposal说的activation则可以指隐藏表示。直接把KL结果等同隐藏层相似度结论，会跳过机制。因此新增目标取最后隐藏层的有效token均值，减去base响应，拟合teacher delta。

隐藏损失按每例teacher delta能量归一化，并设固定floor。两边减同一个base，在损失分子里会抵消，所以不能说这是全新的距离；归一化才明确使用了update能量。mean-pooled hidden也不是模型原生分类pooling，表示更像不保证分类更准。

本轮五种profile全部FP32：同精度CE、hidden-only、CE+hidden、CE+shuffled、global hidden。真实与shuffled使用同一组teacher目标，但后者在任务内打乱输入对应，并保证没有样本仍对到自己。若真实匹配不能稳定优于shuffled，就不能说收益明确依赖逐输入功能信息。若纯hidden把MSE降低却分类更差，说明代理目标与最终任务不完全一致。

本轮新精度与旧v3 BF16拟合不同，所以正确机制对照是新CE FP32 vs新CE+hidden，而不是只拿新hidden与旧BF16 CE比较。所有新增实验在查看v3 final之后提出，仍属于探索性扩展。

## 5. 为什么高router来源识别率不够

来源router回答“这条像SST-2还是MRPC”，而我们真正关心“哪个adapter对这条分类更好”。格式、句子数量和语料风格可能透露来源；高来源accuracy不证明它知道哪个expert正确。

更关键的是，当前每条输入已经由task ID选择分类head。既然task ID已知，直接选本任务adapter本来就能做，且只需一次backbone前向。必须把它列为参照，不能只与random比就宣布router有价值。

新增expert router在fit上用每个expert的GT NLL决定最佳expert目标，再从base feature和已知task one-hot预测选择。它显式增加fit类标签和离线所有expert前向的信息/成本。final oracle则直接看最终GT选真类概率最高的expert，只能当accuracy ceiling，绝不能当可部署方法。报告会分别看实际任务质量、oracle差距及相对own expert的helps/harms。

## 6. 怎样读主指标和最差任务

主指标是SST-2/RTE accuracy、MRPC (accuracy+F1)/2、CoLA MCC的四任务均值。它不是完整官方GLUE score，0.7132也不能说成“所有任务准确率71.32%”。CE实际mean accuracy为78.48%，individual为82.52%。

Worst-task accuracy先在每个seed的四任务里取最小值，再跨seed计算mean±sample SD。CE为68.75%，individual为74.87%。报告同时保留逐任务F1/MCC，因为CoLA有类别不平衡，accuracy可能掩盖问题。

误差条是三次训练的sample SD，不是测试样本confidence interval，也不证明差异显著。三个seed共用同一数据划分，四任务的六个pair还共享adapter，因此不能把18个pair×seed当18次独立实验。

## 7. 数据隔离改善了什么，没有抹去什么

新v3从原train按内容tuple分出训练、teacher选择、系数fit、系数selection和final五区。Teacher重新训练，因为旧teacher已经用过整个calibration选checkpoint，事后再切半不能消除那条信息路径。

新final仍来自研究过的GLUE任务；SST-2的近重复、句对中单句重用、预训练污染都没有被完整tuple去重全部解决。正确说法是“本轮流程内隔离的最终区”，不是“全新外部盲测”，也不是官方validation。新扩展更不能抹去它在设计前已经查看过v3分数这一事实。

## 8. 成本和存储为什么要改口径

原proposal认为多个adapter部署昂贵，但我们的实际bank共享一份base，并不是四套完整backbone。Base约1.84GiB，四个adapter文件约8.33MiB，所以静态合并并没有相对这个bank压缩四倍总存储。

当前A100实现中，batch1/128tokens的individual约23.5ms，GPU缓存router约49.0ms，CPU dense-delta切换约98.6ms。额外base特征前向无法被GPU缓存消除；一批分给多个expert时，前向数是1+不同expert组数。它是我们的实现实测，不是所有优化LoRA serving引擎的通用结论。

97个标量是额外拟合参数；部署还要backbone、任务head以及权重/adapter表示。把它说成“整个模型388bytes”会严重误导。

## 9. Paper联系与Part1的边界

AdaMerging已有逐层系数，RegMean已有response/Gram统计指导合并，KnOTS直接研究LoRA表示对齐，FitNets早已使用hidden hints。我们应说明具体设置、目标和权限的区别，而不是把“activation-aware”这个名称当创新证明。

Part1的CSA帮助理解结构化表示和资源预算；当前Part2代码没有用CSA的mode投影、交替特征分解或CSA重建目标。对教授可以说“conceptual motivation”，不能说“we implemented CSA inside the LLM”。

## 10. 你负责讲代码时，最值得熟悉的五个入口

1. `common.selected_raw`：manifest SHA、dataset fingerprint、固定IDs，回答数据从哪里来。
2. `train_lora.py`：LoRA+head训练与专属teacher选择，回答哪些参数学习。
3. `coefficient_merge/core.py`：softmax/gain和MixedLinear，回答97参数及数学等价。
4. `merge_study/engine.py`：不可变base快照重建状态、单独换head，回答是否累计叠加错权重。
5. `activation_extension.py`：hidden loss、shuffle、router目标、锁定选择后final，以及梯度/数值/ID检查。

更多逐问英文短答见 [实现Q&A](../PART2-IMPLEMENTATION-QA.zh-CN.md)。新实验执行后，应把实际数值补入讲稿；任何尚未核验的内容保留“待核对”，不提前写成优势。
