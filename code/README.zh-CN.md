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

`activation_extension.py` 与 `test_activation_extension.py` 补原proposal的直接hidden匹配、打乱配对、global/layer和best-expert路由对照；复用v3隔离teacher，但新增方法在已查看final后设计，属于探索扩展。使用 [强化配置](../configs/activation-strengthening-v1.json)，说明见 [实验映射](../report-20261109/EXPERIMENT-AND-CLAIM-MAP.zh-CN.md)。

新独立协议入口：`independent_protocol.py` / `test_independent_protocol.py` 负责内容去重、分区与防泄漏检查，`run_independent_study.py` 负责三 seed 重训与 CE/KL，`independent_baselines.py` 重跑固定基线，`summarize_independent_study.py` 核验并输出统计与图表。按 [v3 协议](../INDEPENDENT-PROTOCOL-V3.zh-CN.md) 在 Colab 执行。

`restore_independent_backup.py` 将完成后的 checkpoint ZIP 校验恢复到新目录，重建运行布局并迁移 manifest 路径；遇到已有目录或校验失败会拒绝操作。

部署测量 `benchmark_deployment.py` 与冻结路由标题诊断 `diagnose_router_headers.py` 均要求 CUDA，在 Colab 执行。前者记录逐次耗时、前向次数和缓存，后者保持内容 token、仅改变字段标题。命令与结果解释见 [A100 补充报告](../PART2-FOLLOWUP-2026-11-09.zh-CN.md)。

新任务涉及 FIELDS、selected_raw、两类头部、metrics 和各原型 T=4 假设；换 backbone 还涉及 score head、模块名与层索引。新增合并方法要同时接入 CLI、engine 和 plan。详见中央扩展指南，勿只改 tasks/model 后跑完整实验。子目录都有独立中文说明。

第二轮强化：`robustness_controls.py` 完成selection-only权重/保守gate选择、回退重载和去完整tuple重叠的官方validation复评；`summarize_robustness.py` 独立重算指标并生成表图。原activation配置默认行为保留，新配置才启用逐profile权重与缓存。具体预算与限制见 [协议](../report-20261109/ROBUSTNESS-PROTOCOL.zh-CN.md)。

Direct CSA bridge / CSA直接迁移：[protocol and commands](../report-20261109/CSA-BRIDGE-PROTOCOL.zh-CN.md)、[results and 12 Q&A](../report-20261109/CSA-BRIDGE-RESULTS.zh-CN.md)。`code/csa_bridge.py`直接导入Part1 TensorCSA；`configs/csa-bridge-v1.json`固定网格；新源码/数学检查已纳入双语notebook。

Regularized discriminant follow-up / CSA判别强化：[protocol](../report-20261109/CSA-DISCRIMINANT-PROTOCOL.zh-CN.md)、[results and QA](../report-20261109/CSA-DISCRIMINANT-RESULTS.zh-CN.md)。新增`csa_discriminant.py`、五项测试、独立候选重载审计；所有表示相同网格，中英notebook已同步。
