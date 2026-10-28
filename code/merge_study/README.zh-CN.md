# 合并基线与任务对诊断

[原有说明](README.md) · [中文首页](../../README.zh-CN.md) · [扩展开发](../../docs/EXTENDING.zh-CN.md)

本模块使用已有 bank，不重训 adapter。engine.py 每次从 immutable base+delta 重建参数并使用原任务头，防止候选切换产生残差漂移。

## 单 seed 运行

```bash
python code/merge_study/run.py --config configs/protocol-v2-a100.json --plan configs/merge-study-v2.json --bank outputs/bank-v2-a100-s42 --output outputs/my-merge-study-s42
```

需要 bank 内四任务 adapter/head、状态与元数据。status.json 记录进度，selection.json 保存先锁定的选择，summary.json 保存完成结果。plot.py、aggregate.py 提供图表/多 seed 汇总；它们有各自 CLI，使用 --help 核对参数。

## 对照与扩展

新增方法同时改 merge.py、候选传递和新 plan，不覆盖原网格。主指标由 engine.task_score 定义；增加任务或改类别后核对分母、指标和任务头。TIES 是每矩阵版本，DARE 与 TA 组合，勿换了变体仍沿用原标签。

六任务对共享四个任务，相关系数只作描述。新增 seed 先固定方案，保留逐 seed 的相关性；不能把 18 个重叠 pair 当成 18 个独立重复。test_engine.py 需要固定 transformers 环境的微型模型，用于核验状态和 logits。
