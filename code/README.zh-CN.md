# Part 2 代码入口

[原有说明](README.md) · [中文首页](../README.zh-CN.md) · [扩展开发](../docs/EXTENDING.zh-CN.md)

本目录负责训练、评价、材料化 ΔW 合并、共同输入 probe 及各研究原型。默认使用 Qwen2.5-0.5B、q/v LoRA 与四个二分类任务；不是任意任务的通用注册框架。

## 从哪一个脚本开始

| 文件 | 输入与输出 |
|---|---|
| common.py | 读配置、加载数据/模型、编码、数据 manifest、指标与环境记录 |
| train_lora.py | 单任务训练；保存 adapter、分类头与运行记录 |
| run_bank.py | 调度一个 seed 的四任务训练、individual/average 评价及 probe |
| merge.py | 从 LoRA factor 得到 ΔW，再做 average/TA/TIES/DARE |
| evaluate.py | 原任务头下的 individual 或 merged backbone 评价 |
| probe.py | 共同 calibration 输入上收集 raw 与 adapter−base 表示 |
| run_diagnostics.py | 小样本拟合、head-only 等学习诊断 |

所有命令从仓库根目录执行。完整训练见中文 BUILD；先做小检查：

```bash
python -m unittest discover -s code -p test_math.py
python -m unittest discover -s code -p test_protocol.py
python code/train_lora.py --config configs/protocol-v2-a100.json --task sst2 --seed 42 --max-steps 3 --output outputs/my-smoke-s42
```

最后一条会实际训练，需依赖、下载和资源；3 步不能作为正式精度结论。输出目录必须新建。

## 扩展前看这里

新任务涉及 FIELDS、selected_raw、两类头部、metrics 和各原型 T=4 假设；换 backbone 还涉及 score head、模块名与层索引。新增合并方法要同时接入 CLI、engine 和 plan。详见中央扩展指南，勿只改 tasks/model 后跑完整实验。子目录都有独立中文说明。
