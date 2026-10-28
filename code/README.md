# Part 2 实验代码

[中文使用指南](README.zh-CN.md)

范围：Qwen2.5-0.5B + GLUE（SST-2 / MRPC / RTE / CoLA），q/v LoRA。

- [functional_merge/](functional_merge/README.md)：task-conditioned 线性响应匹配与解析退化控制。
- [coefficient_merge/](coefficient_merge/README.md)：冻结所有模型参数，只拟合全局/逐层混合系数；CE、teacher-KL 与同容量对照。
- [router_v1/](router_v1/README.md)：输入特征、format/random 与 task-ID 路由参照；明确额外前向成本和已知任务头限制。
- [selection_control/](selection_control/README.md)：复用系数历史，检查 teacher-KL 选择能否减少对额外 selection 标签的依赖。
- [transfer_check/](transfer_check/README.md)：固定已有五种配置，在首次接触的 IMDb / PAWS 子集检查迁移，不以目标标签重选模型。

- `common.py`：配置、数据划分、模型、指标及资源记录。
- `train_lora.py`：单任务训练、checkpoint 与断点恢复。
- `evaluate.py`：individual / merged backbone + 任务专属分类头评估。
- `merge.py`：实际 ΔW 的 averaging / task arithmetic / TIES / DARE。
- `probe.py`：共同输入上收集 adapter − base 的 pooled hidden states。
- `test_math.py`：合并数学与 Δactivation 信号的离线检查。
- `test_integration.py`：随机微型 Qwen 的训练、保存、加载、合并及 pooling 端到端检查。

- `run_diagnostics.py`：32 样本拟合、冻结 backbone 对照与小样本 LoRA 学习检查。
- `run_bank.py`：单 seed 的四任务训练、统一 float32 individual/average 评估和共同输入 probe；逐项记录状态，拒绝覆盖已有目录。
- `test_protocol.py`：句对角色边界、长输入截断、旧协议兼容检查。
- [merge_study/](merge_study/README.md)：calibration 选择 Task Arithmetic / TIES / DARE 参数，再评估完整 validation 和六个等权任务对；包含评估器状态与 logits 核验、描述性相关分析和绘图。

Protocol v2 由配置显式启用：单句 Text 字段、句对 Sentence A/B 字段，长句对按内容长度截断并保留两个角色；不写入任务名或标签。训练脚本支持独立 calibration 选最佳 checkpoint、分类头独立学习率、训练/选择样本 ID 清单、学习前后诊断与 `--head-only` 控制。旧配置未指定 input_format 时保持 legacy 行为。

在项目根目录运行脚本，详见 [执行说明](../BUILD.md)。
