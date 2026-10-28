# 组内协作与 Colab

## 平时怎么合作

接受 repo 邀请后，大家用自己的账号读写代码。较大的改动可以先开分支/PR 讨论，小修改按组内习惯来。改动前同步一下远端，有冲突时一起确认，不覆盖别人的工作即可。

当前带团队 Colab 与中文指南的版本在 `feat/team-colab`。代码扩展可以参考 [中文指南](README.zh-CN.md) 和 [开发参考](docs/EXTENDING.zh-CN.md)。

## 第一次用 Colab

1. 打开 [中文 notebook](notebooks/team_colab.zh-CN.ipynb)，或下载 ipynb 后上传到 Colab；private repo 的首次访问可能需要本人授权。
2. 运行环境和检查 cell，代码会从内嵌快照展开；runtime 不需要 GitHub token、git clone 或 MCP。
3. 只复评时恢复 checkpoint，设置 `SEED`、`EVAL_TASK` 和 `RUN_EVALUATION`；想训练时选择 GPU，再开启 `RUN_TRAINING`。
4. 跑完用 export cell 下载结果，或保存到自己有权限的 Drive 路径。

`MEMBER` 是实验记录标签，填昵称或实验编号即可，不需要真实姓名。修改代码/配置后更新 notebook 快照，以免运行的是旧版本。

## 算力与文件

每个 Colab runtime、GPU 额度和 Drive 权限分别属于各自账号。可以各自跑实验，也可以由有空闲 GPU 的组员集中执行，其他人提供代码、配置或分析；不需要为了协作共用账号。

[已有 checkpoint 清单](assets/checkpoints-2026-10-25.json) 对应约 24.6 MB 的独立包，含 12 个 adapter 和相关历史。下载位置由组内另行约定；拿到包后上传或填写 Drive 路径即可恢复，函数会校验内容。GitHub 权限不会自动变成 Drive 权限。

临时 VM 中的结果可能消失，重要实验记得导出。小结果 JSON/图表可以放 Git，较大的 checkpoint、缓存和日志单独共享。新 run 用不同名字，方便查找和对比。

官方操作说明可查 [Colab FAQ](https://research.google.com/colaboratory/faq.html)。当前恢复/导出已做本地验证，组员首次授权和 GPU 分配仍需在自己的账号上确认。
