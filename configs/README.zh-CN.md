# 配置与实验方案

[原有说明](README.md) · [中文首页](../README.zh-CN.md) · [扩展开发](../docs/EXTENDING.zh-CN.md)

这里同时存两类 JSON：模型/训练 cfg 与研究选择 plan。Runner 的 --config 通常指前者，--plan 指后者，二者不能互换。

## 配置怎么选

`activation-strengthening-v1.json` 用于原proposal强化：五种FP32系数目标、固定hidden权重/能量floor、source/expert router候选；输入为匹配v3 manifest的bank，属于探索性扩展，见 [主张映射](../report-20261109/EXPERIMENT-AND-CLAIM-MAP.zh-CN.md)。

新实验采用 `independent-v3.json`（teacher）与 `independent-coefficients-v3.json`（同容量 CE/KL）。编排器生成固定版本数据的内容分组 manifest，配置锁定 SHA；旧配置的 v2 行为保持兼容。见 [独立协议](../INDEPENDENT-PROTOCOL-V3.zh-CN.md)。

| 文件 | 用途 |
|---|---|
| pilot.json | 最早的低预算/旧输入协议记录 |
| protocol-v2.json、protocol-v2-a100.json | 显式句对输入、训练/calibration 划分与 v2 预算；A100 版 batch16/累积1 |
| merge-study-v2.json、merge-study-replication-v2.json | 17 候选合并网格、选择规则与重复研究 |
| functional-v1.json | 响应统计、simplex、beta 与 gain 候选 |
| coefficient-v1.json、coefficient-ce-control.json | fit/select 样本、steps、CE/KL profile 与同容量控制 |
| router-v1.json | 特征路由拟合、选择和对照 |
| selection-control.json、transfer-check.json | 无额外选择标签的后续控制、固定迁移检查 |

## 新配置的正确做法

复制为新名字并记录协议版本，不覆盖归档配置。检查新键确实被脚本读取：例如部分 run_all.py 固定 seeds 42/43/44 和输出路径，JSON 写入新 seeds 并不会自动改变所有脚本。coefficient 的 24 层、四任务维数与二分类也不是模型名字变化后自动适配。

先运行完整仓库检查，再按目标 CLI 用新配置和新输出目录做小检查。runtime 配置变化后运行 scripts/build_team_notebook.py 更新中英 launcher。候选、选择集、tie-break 与预算在查看目标结果前固定，具体例子见扩展指南。
