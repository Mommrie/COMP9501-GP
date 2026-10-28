# 选择阶段标签控制

[原有说明](README.md) · [中文首页](../../README.zh-CN.md) · [扩展开发](../../docs/EXTENDING.zh-CN.md)

复用已有 global/layer KL 系数 history，改用 selection 半集的 teacher-KL 选择 step/gain；不新增训练，也不把目标 validation 分数用于改选候选。

```bash
python code/selection_control/run.py
```

脚本当前没有通用路径 CLI，固定读取 outputs/coefficient-v1、三个标准 bank，并创建 outputs/selection-control。恢复包若已有这个目录，保留结果；重新研究时先增加路径参数或使用新的完整工作副本。

## 输出与扩展

每个 seed 保存锁定选择及候选 KL，再做最终 validation。温度、候选 steps 和 uniform gains 在 plan 中固定。增加新选择准则时匹配候选轨迹，明确哪些标签进入 fit/selection/teacher。

“没有额外 selection 标签”仅描述这一准则；原 teacher/head 训练和 teacher checkpoint 选择使用了标签，且 validation 已用于前期探索。不能称为完全无监督或全新盲测。
