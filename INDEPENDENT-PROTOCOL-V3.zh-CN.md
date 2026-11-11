# Part 2：独立数据协议 v3

冻结日期：2026-11-09。新实验在 Colab A100 上执行；旧实验结果保持不变。此文描述设计与权限，完成状态和实际指标另见结果报告。

执行已完成：[三 seed 结果、图表与展示材料](PART2-RESULTS-V3-2026-11-09.zh-CN.md)，[恢复验证](artifacts/independent-v3-restore-2026-11-09.json)。

本轮数据版本：`nyu-mll/glue` commit `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`。首次启动解析后立即写入 locked-config；默认模板也补入相同 commit，后续复现不随 Hub main 漂移。

## 要回答的问题

在相同可训练容量、相同拟合输入、相同步数、相同初始化、相同 selection 标签权限下，individual teacher 的软输出是否比直接 CE 监督更适合学习静态合并系数？同时比较固定的权重空间基线。

这是原 proposal 的一个可检验子问题。KL 匹配的是最终分类 logits，不是直接匹配所有隐藏层 activation；不能把 KL 胜出自动解释为中间 activation 相似度假设成立。研究完整原始假设的旧 pair 相似度结果仍需单独引用。

## 五个信息分区

全部从固定版本的 GLUE 原 train 产生，不加载原 validation。对字段文本做 NFKC、casefold、空白规范化后生成 tuple hash；相同完整字段 tuple 在不同任务中也分到同一信息分区。每个 tuple 仅保留一个代表样本。根据固定 seed 与内容 hash 分配，不查看 label；不是按标签分层采样。

| 内部 role | 用途 | hash 区间比例 | 每任务上限 |
|---|---|---:|---:|
| train | 训练新 LoRA 与任务头 | 65% | 4096 |
| calibration | teacher 每 epoch checkpoint 选择 | 8% | 128 |
| coefficient_fit | CE 或 KL 系数更新 | 8% | 128 |
| coefficient_selection | 系数 step、uniform gain 与基线候选选择 | 8% | 128 |
| validation | 本轮最终评估；代码兼容名，实际来源为原 train | 11% | 256 |

上限之外的样本不用；比例是 hash 区间，实际数量不是精确比例。每任务实际训练量：SST-2 4096、MRPC 2391、RTE 1571、CoLA 4096，其余四个 role 均达到表中上限。相同句对的一句话重复、反向句对、近似重复不属于此处完整 tuple 去重的覆盖范围；不能称排除了所有内容泄漏。

`partitions.json` 保留完整 sample IDs、content hash、源 fingerprint 和 Hub revision。锁定配置记录 manifest SHA256；读取前复核 SHA 与原始 dataset fingerprint，拒绝静默改分区。`validate_manifest` 检查任务内 IDs、跨任务完整内容 hash 的角色一致性。最终集与前四区分离，但它仍来自曾研究过的 GLUE 任务；这提供本轮流程隔离，不提供“从未受此前研究影响的全新外部盲测”保证。

## Teacher 与合并

每 seed 42/43/44 重新训练四个 adapter，不恢复旧 teacher。固定 Qwen2.5-0.5B commit、r=8、alpha=16、q/v projection、128 tokens、batch 16、3 epochs，teacher 根据专属 calibration loss 选择 epoch。任务头一直 task-aware，永不合并；直接按 task ID 选择 individual 是可用参照。

合并使用实际更新矩阵 `ΔW=(alpha/r)BA`。不能直接把各 adapter 的 A/B 分别平均，因为会引入跨 adapter 乘积项。学生每层采用四个任务的 softmax 权重和共享正 gain：

`W_l = W_base,l + gain * sum_t softmax(a_l)[t] * ΔW_l,t`

24 层×4 logits 加 1 gain，共 97 个可训练标量；softmax 存在平移冗余，所以这是参数张量计数，并非 97 个独立自由度。base、LoRA A/B、各任务头均冻结。拟合用 MixedLinear 的低秩响应，最后物化为单模型，检查与 dense merge logits 一致。

CE 与 KL 都训练 120 steps、batch 8、lr 0.03、相同 penalty，任务循环顺序和抽样 generator 相同。KL 温度为 2，使用 individual teacher 在 fit 区上的 soft logits，损失乘 T²。selection 候选 steps 为 0/30/60/90/120，以相同 GT macro native score 选择，平局保留更早 step；因此 KL 的系数梯度不使用 fit GT，但整个流程有 teacher 训练和 selection 标签，不能称无监督。

uniform gain 候选为 1、2。权重空间基线复用原 17 项固定网格，在 coefficient_selection 的同样 128 条/任务上选择 Average、TA、TIES、DARE；不同方法候选数量不同，不声称选择计算预算完全相同。TIES 是每矩阵 top-k、幅度投票符号与 disjoint mean；DARE 是固定 mask seed 42 的 drop/rescale+TA。本轮不是对官方实现所有变体的穷举复现。

每 seed、每种方法先保存选择文件，再加载该方法最终评估分区。不能根据最终分数挑选新超参后继续把同一分区称未接触 holdout。三 seed 使用同一数据划分，量化训练 seed 差异，不是三次独立数据集试验。

## 指标和结论范围

- 逐任务保留 accuracy、F1、MCC。主指标：SST-2/RTE accuracy、MRPC (accuracy+F1)/2、CoLA MCC；报告四任务均值与 sample SD，不称官方完整 GLUE 分数。
- 另外报告 mean accuracy、worst-task accuracy 与相对同 seed individual 的逐任务 accuracy drop。异尺度 native score 的最小值不宜直接称最差任务能力。
- CE/KL 使用同 seed 配对差值，列出全部三点；n=3 不做可靠显著性或大范围泛化保证。
- 新协议改变训练数据量和划分，不能把与旧 v2 的差异归因于单个算法改动。
- 已知任务头条件下，router 不能替代直接 task-ID 选择的优势论证；其成本与标题敏感性作为诊断材料保留。

## 执行与复现

仅在已准备依赖的 Colab A100、仓库根目录执行，输出目录必须新建：

```bash
export USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
python code/run_independent_study.py --output outputs/my-independent-v3
python code/independent_baselines.py --study outputs/my-independent-v3
```

配置见 [teacher 配置](configs/independent-v3.json)、[CE/KL 对照](configs/independent-coefficients-v3.json)。[分区模块](code/independent_protocol.py) 生成清单，[公共数据入口](code/common.py) 校验配置锁，[系数学习](code/coefficient_merge/run.py) 拒绝旧 teacher，[编排器](code/run_independent_study.py) 保留日志与源码 SHA。所有权重、checkpoint 和日志先留云端，完成后按 manifest 导出备份。

恢复本轮 checkpoint ZIP 时使用 [专属恢复器](code/restore_independent_backup.py)，输出目录必须不存在：

```bash
python code/restore_independent_backup.py \
  --archive /content/COMP9501-Part2-independent-v3-20261109-checkpoints.zip \
  --output outputs/restored-independent-v3
```

恢复器先检查 CRC、全部文件 SHA 和路径，再重建 bank 与系数结果布局，迁移 locked-config 的 manifest 路径；原始 evidence 保持不变。它不重复打包或恢复 Hub base/tokenizer，不恢复优化器状态，也不将恢复操作说成重新训练。后续复评使用新目录中的 locked-config 与相应 bank；训练编排器拒绝覆盖该目录。
