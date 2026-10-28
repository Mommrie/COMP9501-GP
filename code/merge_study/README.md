# 合并基线与六任务对诊断

[中文使用指南](README.zh-CN.md)

运行入口为 `run.py`，仅使用已训练好的 protocol-v2 adapter 银行，不再训练模型。`engine.py` 在一个 float32 base model 上从保存的原始参数重建每个合并状态，并装入该任务原有分类头，避免连续加减造成漂移。`test_engine.py` 检查状态切换与任务头，并将真实模型输出与既有评估器交叉核验。

参数网格和诊断方向先写入 [预定方案](../../configs/merge-study-v2.json)。只用 calibration 选择各方法参数，将选择结果写入磁盘后才评估完整 validation。TIES 沿用已有的每矩阵 top-k 版本；DARE 与 Task Arithmetic 组合。Task Arithmetic 系数 1/4 等于四任务平均。

六个任务对分别做等权 ΔW 合并，再在各自任务上使用原分类头。核心描述性检验是共同输入上的 Δactivation cosine 与平均 accuracy 损失的相关性（假设负相关）；另看任务主指标、raw cosine、权重 cosine 和各 probe 来源子集。六个任务对共享四个任务，不能当作六个独立样本，单 seed 不报告显著性结论。

从项目根目录执行：

```bash
python code/merge_study/test_engine.py
python code/merge_study/run.py --config configs/protocol-v2-a100.json --plan configs/merge-study-v2.json --bank outputs/bank-v2-a100-s42 --output outputs/merge-study-s42
```

输出目录必须不存在。状态和逐步结果保存在 `status.json`，参数锁定在 `selection.json`；全部运行完成后输出 `summary.json`。绘图入口为 `plot.py`。

`replicate.py` 为 seeds 43/44 启动各自独立输出目录，先运行 `run_bank.py` 再运行同一 study；可用两个进程共享 A100，主机训练线程各限为 4。复现配置为 [merge-study-replication-v2.json](../../configs/merge-study-replication-v2.json)，保留原参数网格，DARE mask seed 固定 42，以分离训练 seed 的变化。原 seed 42 银行和结果均不覆盖。

`aggregate.py` 校验三次运行的配置、网格与逐 seed 相关性，输出 mean / sample SD（ddof=1）、每次选择的参数、六个唯一任务对的汇总及 PNG/PDF。它不生成 p 值，也不把 18 个重叠任务对当成独立样本。示例（输出目录选新的位置）：

```bash
python code/merge_study/replicate.py --config configs/protocol-v2-a100.json --plan configs/merge-study-replication-v2.json --output outputs/replication-s43-s44 --seeds 43 44 --workers 2
python code/merge_study/aggregate.py --studies artifacts/merge-study-summary-2026-10-25.json artifacts/replication-s43-study-2026-10-25.json artifacts/replication-s44-study-2026-10-25.json --output outputs/review/summary.json --plot outputs/review/diagnostics
```
