# 实验配置

[中文使用指南](README.zh-CN.md)

`functional-v1.json`、`coefficient-v1.json`、`coefficient-ce-control.json` 和 `router-v1.json` 为本轮静态/动态原型的冻结方案；对应报告见 [PROTOTYPES](../PROTOTYPES-2026-10-25.md)。所有新方法 validation 均属前期基准后的探索，不是全新确认集。

`merge-study-v2.json` 固定合并诊断的 17 组配置（其中 TA 0.25 与 average 等价）：Task Arithmetic 系数、TIES 每矩阵保留率/系数、DARE 保留率/系数。最多 256 条 calibration 按四任务主指标均值选参数，锁定后才评估 validation；六任务对相关性只作单 seed 描述。方案和源码指纹先归档到 artifacts。

`merge-study-replication-v2.json` 用于训练 seeds 43/44 的重复，网格和诊断规则与 seed 42 一致，仅更新说明文字；各 seed 单独按 calibration 选择参数。DARE mask seed 继续固定 42。

`pilot.json` 固定初步构建的模型、LoRA、训练及数据划分。JSON 可由 Python 标准库直接读取。

`activation-strengthening-v1.json`：在v3 bank上固定五个FP32系数profile、hidden能量归一化、打乱响应及性能导向路由控制；必须将其结果作为已查看v3 final之后的探索扩展，不能改名成新盲测。见 [实验映射](../report-20261109/EXPERIMENT-AND-CLAIM-MAP.zh-CN.md)。

`independent-v3.json` 与 `independent-coefficients-v3.json` 固定新 teacher 训练和同容量 CE/KL 对照，运行时生成内容分组的五分区 manifest 并锁定 SHA256。详见 [独立协议](../INDEPENDENT-PROTOCOL-V3.zh-CN.md)，不恢复旧 teacher。
pilot 只用于验证流水线，正式实验需要扩大训练量并运行 3 个 seeds。

`protocol-v2.json` 是下一阶段起始方案：显式句对边界、最长 128 tokens、每任务训练最多 4096 条（不足则取可用部分）、3 epochs、有效 batch 16；分类头 lr=1e-3、LoRA lr=2e-4、10% warmup。仅用原 train 内 calibration（最多 256 条）选 checkpoint；eval_samples=0 表示另行评估完整 GLUE validation。此配置是待测起点，不是已证实最优参数。诊断运行会生成独立的小预算配置并记录，不修改 pilot.json。

- `protocol-v2-a100.json`：v2 的 A100 配置，batch 16 / 累积 1，有效 batch 仍为 16，固定已验证的 Qwen commit。用于 seed 42 扩展试验，仍需至少 3 seeds；不要与 v1 结果直接作归因比较。
# 后续选择标签控制

`selection-control.json`：固定原 KL 系数历史与统一 gain 候选，以 held-out teacher KL 选择，不新增训练；teacher 的既有标签使用与 validation 的探索性边界保留。
# 外部数据迁移检查

`transfer-check.json`：五种已固定方法、三 adapter seeds，IMDb / PAWS test 各 label-blind 抽 512 条；只复用 SST2 / MRPC 头，不在目标域调参。
# 外部数据迁移检查

`transfer-check.json`：五种已固定方法、三 adapter seeds，IMDb / PAWS test 各 label-blind 抽 512 条；只复用 SST2 / MRPC 头，不在目标域调参。

`activation-sensitivity-v2.json` adds aligned/shuffled weights .01 and .1, reusing completed 0/1 endpoints. `routing-safety-v2.json` freezes fallback thresholds, Ridge grid, selection constraints and cleaned-official evaluation. These are exploratory extensions after earlier final inspection; see the [protocol](../report-20261109/ROBUSTNESS-PROTOCOL.zh-CN.md).

Direct CSA bridge / CSA直接迁移：[protocol and commands](../report-20261109/CSA-BRIDGE-PROTOCOL.zh-CN.md)、[results and 12 Q&A](../report-20261109/CSA-BRIDGE-RESULTS.zh-CN.md)。`code/csa_bridge.py`直接导入Part1 TensorCSA；`configs/csa-bridge-v1.json`固定网格；新源码/数学检查已纳入双语notebook。
