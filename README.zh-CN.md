# 项目中文首页

[原有说明](README.md) · [扩展开发](docs/EXTENDING.zh-CN.md)

这里存放 COMP9501 Group 13 的共享代码与实验记录，大家可以按自己负责的部分使用和补充。Part 1 是 CSA 论文讲解与二维补充复现，Part 2 是 LoRA adapter 合并研究。

项目想做什么、目前做到哪里，先看 [Proposal 与进度清单](PROPOSAL.md)：保留原始提案正文，逐项对照现有证据，尾部有建议 timeline、已完成和待办事项。

**Part 1 代码与展示证据**：直接看 [CSA 材料包](part1-reproduction-20261025/presentation-20261028/README.md)，含公式对应、代码讲解、10 组图表、实验解释和 Q&A；可交给 PPT 制作者使用，CPU 即可复现。

## 第一次使用怎么走

1. 阅读 [中文搭建与复现](BUILD.zh-CN.md)。想直接跑 Colab，可打开 [中文团队 notebook](notebooks/team_colab.zh-CN.ipynb)；private repo 的首次访问仍需本人授权。
2. 只复评时恢复共享 checkpoint，使用 notebook 的 RUN_EVALUATION；要新训练时选 GPU、设置自己的 MEMBER/SEED，并使用新的 run 目录。
3. 运行完成后导出带 source snapshot、seed 和文件 SHA 的结果包；数据、权重、Drive 权限与 GitHub 代码分开管理。
4. 开发时可以参考 [组员提交指南](CONTRIBUTING.zh-CN.md) 与 [扩展指南](docs/EXTENDING.zh-CN.md)，较大改动可开分支/PR 讨论。

当前团队功能在 feat/team-colab，尚未合入 main。Notebook 内嵌源码包，精确大小和 SHA 见 [快照记录](artifacts/team-launcher-2026-10-28.json)，不需 runtime GitHub token 或 MCP。共享 notebook 不共享其他账号的 GPU 额度或临时 VM，见 [协作说明](COLLABORATION.md)。

## 各板块中文入口

| 板块 | 中文说明 |
|---|---|
| 训练、评价、合并与 probe 总入口 | [code](code/README.zh-CN.md) |
| 合并基线与任务对诊断 | [merge_study](code/merge_study/README.zh-CN.md) |
| 局部响应匹配 | [functional_merge](code/functional_merge/README.zh-CN.md) |
| CE/KL 静态系数 | [coefficient_merge](code/coefficient_merge/README.zh-CN.md) |
| 输入路由 | [router_v1](code/router_v1/README.zh-CN.md) |
| 选择标签控制 | [selection_control](code/selection_control/README.zh-CN.md) |
| 外部迁移检查 | [transfer_check](code/transfer_check/README.zh-CN.md) |
| 配置与实验方案 | [configs](configs/README.zh-CN.md) |
| Colab notebook | [notebooks](notebooks/README.zh-CN.md) |
| 打包、恢复与检查工具 | [scripts](scripts/README.zh-CN.md) |
| 权重共享清单 | [assets](assets/README.zh-CN.md) |
| 科学证据与结果 | [artifacts](artifacts/README.zh-CN.md) |
| CSA 演示、代码与结果 | [Part 1](part1-reproduction-20261025/README.zh-CN.md)、[代码](part1-reproduction-20261025/code/README.zh-CN.md)、[证据](part1-reproduction-20261025/artifacts/README.zh-CN.md) |
| 扩展开发、计划与 PR 模板 | [docs](docs/README.zh-CN.md) |

## 现有结果如何理解

四任务为 SST-2/MRPC/RTE/CoLA，使用已知的原任务分类头。四任务主指标均值是 SST2 acc、MRPC (acc+F1)/2、RTE acc、CoLA MCC 的均值，不是官方完整 GLUE 分数。三 seed 的 individual 约 0.7684，逐层 CE 约 0.6997，teacher-KL 选择的逐层 KL 约 0.6996；尚未建立 activation 信号的独特优势或稳定 OOD 迁移。

完整分析见 [三 seed](REPLICATION-2026-10-25.md)、[方法原型](PROTOTYPES-2026-10-25.md)、[选择控制](SELECTION-CONTROL-2026-10-25.md)、[迁移检查](TRANSFER-CHECK-2026-10-25.md)。已有 validation 被多次用于探索，teacher 训练/选择用过标签；请在新扩展中保留这些边界。

## 队友想扩展项目

可以先做独立配置消融，再扩展方法、数据或模型。当前四任务/二分类/24 层等固定假设已逐项列在 [扩展开发指南](docs/EXTENDING.zh-CN.md)，并配有最小实验示例、修改点和验证清单。新结果放新目录，旧证据、真实贡献和引用保留。
