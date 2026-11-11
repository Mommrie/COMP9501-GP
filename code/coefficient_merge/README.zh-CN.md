# 端到端静态系数

新 [独立协议 v3](../../INDEPENDENT-PROTOCOL-V3.zh-CN.md) 在配置中启用 `partition_manifest`，读取独立 fit/selection 分区并校验新 teacher 的 manifest SHA；不启用时保留原探索性协议。

[原有说明](README.md) · [中文首页](../../README.zh-CN.md) · [扩展开发](../../docs/EXTENDING.zh-CN.md)

冻结 base、LoRA A/B 与原任务头，只学习 softmax 混合系数和公共 gain。CE 使用 fit 标签，KL 使用 individual teacher logits；原版 checkpoint 选择使用后半区标签。global 为 5 个、24 层版本为 97 个可训练标量。

```bash
python code/coefficient_merge/run.py --config configs/protocol-v2-a100.json --plan configs/coefficient-v1.json --bank outputs/bank-v2-a100-s42 --output outputs/my-coeff-s42
python -m unittest discover -s code/coefficient_merge -p test_core.py
```

先有 bank 才能取得 teacher、LoRA factors 和 heads。输出包含 fit/select ID、teacher logits、候选 history、锁定 selection 与 validation。同容量 layer CE 用 coefficient-ce-control.json；批量 runner 的原目录固定，另开实验时检查路径。

## 后续扩展

core.py 里任务数写死 4，plan 里逐层数为 24，层号解析及 score head 也与当前模型关联。扩展时核对冻结参数梯度、初始 logits 和材料化 delta 等价性；保留 uniform、step0、相同容量/步数对照。

Teacher 原先用过全 calibration 选 checkpoint；本次 fit/select ID 分开不等于全流程独立 holdout。相关边界不能随着修改目标函数而省略。
