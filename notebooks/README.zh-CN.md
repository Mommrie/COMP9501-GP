# Colab 使用入口

[原有说明](README.md) · [中文首页](../README.zh-CN.md) · [扩展开发](../docs/EXTENDING.zh-CN.md)

优先使用 [team_colab.zh-CN.ipynb](team_colab.zh-CN.ipynb)。它与 [英文版](team_colab.ipynb) 的代码 cell 完全一致，区别是说明文字；quickstart.ipynb 是需要已有 checkout 的简化入口。

## 一次运行的顺序

1. 从当前公开团队仓库打开 notebook，或下载 ipynb 上传到 Colab；运行时使用自己的 Google 账号。
2. 执行源码校验/展开、依赖安装和离线检查 cell。内嵌固定代码快照，大小与 SHA 见 [快照记录](../artifacts/team-launcher-2026-10-28.json)，不包含原数据或模型权重；Part 1 的新张量实现与方案已同步。
3. 已有 checkpoint 时填 CHECKPOINT_ARCHIVE 恢复；只复评则设置 SEED/EVAL_TASK 并开启 RUN_EVALUATION。
4. 新训练时选 GPU，设置自己的 MEMBER、SEED 与 RUN_TRAINING。每次 run 目录有独立随机标识。
5. 开启 EXPORT_RESULTS，下载结果 ZIP 或选择自己已授权的 Drive 路径，离开临时 VM 前保存。

**Part 2 最终协议 v3**：新增两个可选部分。填写 `INDEPENDENT_CHECKPOINT_ARCHIVE` 可恢复 v3 的十二个 teacher；或开启 `RUN_FRESH_V3` 从头训练并运行固定基线。之后开启 `RUN_ACTIVATION_CONTROLS`，执行五种 FP32 目标和 source/expert router 对照，需要至少 20 GiB 的 CUDA GPU（实测 L4）。多 seed 导出记录会列出 42/43/44；复现已有 seed 不计为新增独立实验。默认不执行这些操作。

默认开关关闭训练/评价/导出，Run all 不会自动启动完整 GPU 实验。私库访问、Google runtime、Drive 权限分别管理；组员不需安装 MCP。

## 修改与排错

不要在生成的 base64 cell 里改源代码。先在仓库分支修改 code/config，再运行 `python scripts/build_team_notebook.py` 重建中英两个文件，用 `--check` 验证同步。runtime 源码有本地改动时，bootstrap 会拒绝用不同内容覆盖。

GPU 额度属于各自账号；可以由有空闲 GPU 的组员集中跑实验、组员提交代码和分析。细节见仓库 COLLABORATION.md。恢复包的共享云位置仍待设置，不能假定拥有 repo 权限就已得到 Drive 权限。

## 维护中英说明

英文说明在生成器中，中文文字在 [team_notebook.zh-CN.json](team_notebook.zh-CN.json)。修改英文说明后，先审阅并更新中文翻译，再用 `python scripts/build_team_notebook.py --print-markdown-digest` 获取英文说明哈希，更新 JSON 的 source_markdown_sha256，最后重建两个 notebook。这样英文说明的变化不会被旧翻译静默覆盖。

Direct CSA bridge / CSA直接迁移：[protocol and commands](../report-20261109/CSA-BRIDGE-PROTOCOL.zh-CN.md)、[results and 12 Q&A](../report-20261109/CSA-BRIDGE-RESULTS.zh-CN.md)。`code/csa_bridge.py`直接导入Part1 TensorCSA；`configs/csa-bridge-v1.json`固定网格；新源码/数学检查已纳入双语notebook。

Regularized discriminant follow-up / CSA判别强化：[protocol](../report-20261109/CSA-DISCRIMINANT-PROTOCOL.zh-CN.md)、[results and QA](../report-20261109/CSA-DISCRIMINANT-RESULTS.zh-CN.md)。新增`csa_discriminant.py`、五项测试、独立候选重载审计；所有表示相同网格，中英notebook已同步。
