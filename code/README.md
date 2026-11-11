# Part 2 实验代码

[中文使用指南](README.zh-CN.md)

范围：Qwen2.5-0.5B + GLUE（SST-2 / MRPC / RTE / CoLA），q/v LoRA。

- `activation_extension.py` / `test_activation_extension.py`：原proposal强化实验，FP32隐藏响应、真实/打乱配对与global/layer控制；另比较source/best-expert router及GT oracle。复用v3 teacher，最终区已有探索历史，见 [实验与主张映射](../report-20261109/EXPERIMENT-AND-CLAIM-MAP.zh-CN.md)。模型要求Colab CUDA GPU至少20GiB；A100未获分配时本轮使用L4，所有新控制保持同硬件。

- 独立协议 v3：`independent_protocol.py` 生成五分区，`test_independent_protocol.py` 检查重复文本与跨任务信息边界；`run_independent_study.py` 依次重训三 seed teacher 并比较同容量 CE/KL；`independent_baselines.py` 运行固定权重基线；`summarize_independent_study.py` 审计、汇总 CSV 和生成图表。模型计算和绘图均放 Colab，详见 [协议](../INDEPENDENT-PROTOCOL-V3.zh-CN.md)。
- `restore_independent_backup.py`：校验 checkpoint ZIP 的 CRC、逐文件 SHA 和路径，恢复到新目录，重建 bank/系数/选择记录并迁移 manifest 路径；不覆盖已有目录。

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

2026-11-09 补充工具（均要求 CUDA，请在 Colab 运行）：

- `benchmark_deployment.py`：固定输入、同步计时，比较静态合并与 CPU/GPU 缓存路由；记录逐次耗时、前向次数与实际缓存占用。
- `diagnose_router_headers.py`：冻结已有 router，比较原标题、去标题和统一标题，保留内容 token 与样本 ID。

命令、统计口径与限制见 [补充报告](../PART2-FOLLOWUP-2026-11-09.zh-CN.md)。

The follow-up uses `activation-sensitivity-v2.json` with optional per-profile hidden weights and role caches. `robustness_controls.py` locks lambda/gates using selection only, checks conservative routing and evaluates content-filtered official validation; `summarize_robustness.py` independently audits saved predictions and generates tables/figures. See [protocol](../report-20261109/ROBUSTNESS-PROTOCOL.zh-CN.md).

Direct CSA bridge / CSA直接迁移：[protocol and commands](../report-20261109/CSA-BRIDGE-PROTOCOL.zh-CN.md)、[results and 12 Q&A](../report-20261109/CSA-BRIDGE-RESULTS.zh-CN.md)。`code/csa_bridge.py`直接导入Part1 TensorCSA；`configs/csa-bridge-v1.json`固定网格；新源码/数学检查已纳入双语notebook。

Regularized discriminant follow-up / CSA判别强化：[protocol](../report-20261109/CSA-DISCRIMINANT-PROTOCOL.zh-CN.md)、[results and QA](../report-20261109/CSA-DISCRIMINANT-RESULTS.zh-CN.md)。新增`csa_discriminant.py`、五项测试、独立候选重载审计；所有表示相同网格，中英notebook已同步。
