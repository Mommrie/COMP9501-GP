# 局部响应匹配

[原有说明](README.md) · [中文首页](../../README.zh-CN.md) · [扩展开发](../../docs/EXTENDING.zh-CN.md)

在各任务自己的 calibration 输入上收集 LoRA q/v 响应 Gram，再拟合 simplex 上的全局/逐层系数；gain 与 uniform 用同样候选选择。maths.py 是优化核心，run.py 连接统计、选择与最终评价。

```bash
python code/functional_merge/run.py --config configs/protocol-v2-a100.json --plan configs/functional-v1.json --bank outputs/bank-v2-a100-s42 --output outputs/my-functional-s42
python -m unittest discover -s code/functional_merge -p test_maths.py
```

需要单 seed 四任务 bank，输出包含统计/系数/选择和验证结果。run_all.py 固定原三 seed 银行路径与输出名称；已有目录时不要直接覆盖。

## 后续扩展

共同输入、同权重的平方匹配包含 uniform 最优解，保留这个解析控制。改输入分布、层或归一化时记录清楚；局部响应目标下降不能当成整个模型分类损失改善。

目前 Gram/系数/若干循环和平均分母写死四任务，层号取自参数名；增加任务或换模型要按中央扩展表修改，并验证 task order、有效 token 掩码和目标维数。
