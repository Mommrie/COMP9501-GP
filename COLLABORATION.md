# 组员协作与 Colab 使用

## GitHub 协作

仓库 owner 邀请组员，组员接受后即可用各自账号读写 private repo。建议 `main` 保持稳定，每项改动用分支 + Pull Request；这是协作约定，目前不声称已配置服务器端分支保护。

```bash
git switch main
git pull --ff-only
git switch -c feat/your-alias-topic
```

提交前运行 `python scripts/check_repository.py` 和受影响的测试。修改运行代码、配置或恢复工具后，再运行 `python scripts/build_team_notebook.py` 更新嵌入快照。每人使用自己的真实 Git 署名；`lrtang` 是本维护者的提交身份，不是要求组员共用的身份。保留真实贡献、引用与课程要求的披露；不添加自动工具联名。

在 Colab 里改过的实验代码要通过自己的分支带回 GitHub。运行快照不自动 push，避免把临时输出或过期代码覆盖到共享仓库。合并前先同步远端，遇到冲突保留双方意图并核对测试，不强推覆盖别人的工作。

## 组员第一次运行

1. 接受 private repo 邀请。
2. 打开 [team_colab.ipynb](notebooks/team_colab.ipynb)。仓库发布后可用下面的 Open in Colab 入口；如私库直达需要授权，按 Colab 提示完成。若暂时打不开，从 GitHub 下载该 notebook，再在 Colab 上传打开也可。
3. 首先 Run all：自动校验并展开约 70 KB 的代码压缩快照、安装固定依赖、运行离线检查。runtime 里不需要 GitHub token、git clone 或 MCP 客户端。公开模型和数据仍通过原发布方下载。
4. 要训练时切换 GPU，设置自己的 `MEMBER`、seed 与 `RUN_TRAINING=True`，再执行相应训练 cell。输出目录有随机标识，不覆盖已有 run。

[Open in Colab](https://colab.research.google.com/github/Mommrie/COMP9501-GP/blob/main/notebooks/team_colab.ipynb)

这个入口是源代码执行快照，不是一个已认证的 Git checkout。启动时显示快照 SHA；checkpoint 和结果记录使用同一 SHA 对齐。若源代码修改后没有重建 launcher，仓库检查会指出快照过期。

## GPU 与账号

GitHub collaborator 权限、Google Drive 文件权限与 Colab 算力分别管理。共享 notebook 不共享已安装库、临时文件、当前 VM 或 GPU 额度；你的 Google AI Pro 权益不会因邀请组员加入 GitHub 而转移给他们。

两种正常工作方式都可以：组员各自用自己的 Colab runtime 跑小实验；或组员提交代码/配置与分析，由维护者在自己的 GPU runtime 集中执行。后一种方式共享实验成果，不共享账号或让多人同时控制一个临时 kernel。组员不必为了查看结果、改代码或运行 CPU 检查而订阅付费 GPU。

## 复用已有权重

已准备 [12 个 adapter 与历史系数的共享清单](assets/checkpoints-2026-10-25.json)。约 24.6 MB 的 ZIP payload 保存在维护者本地，未进 Git，云端下载位置尚未设置。它只含本课程的 adapter、bank 元数据与 coefficient/selection histories，不含 base model、原始数据和其他课程资料。

Owner 可把这个 ZIP 放入 private group Drive 文件夹并单独给组员授权。组员下载后用 Colab Files 上传，或使用自己的已授权 Drive mount 路径，填入 notebook 的 `CHECKPOINT_ARCHIVE`。恢复函数检查整个 ZIP 与每个文件 SHA，拒绝覆盖不同内容；恢复完成即可使用原有 bank 路径做评估，不必先重训 12 个 adapter。

只想复评时，保持 `RUN_TRAINING=False`，设置 `SEED` 和 `EVAL_TASK`，开启 `RUN_EVALUATION` 的 cell。结果写入新的输出目录，可以直接接后面的 export cell。

GitHub 邀请不自动授予 Drive 访问权限。当前不会自动挂载或写入任何人的 Drive；先选择已有共享存储位置再配置下载入口。

## 结果保存与交接

临时 runtime 会消失，重要 run 执行后立即导出。Notebook 的 export cell 为选中的 `outputs/<run>` 打包，写入 member、seed、source snapshot 和逐文件 SHA；可以下载 ZIP，或把 `EXPORT_DIRECTORY` 指到自己已授权的 Drive 文件夹。归档不能覆盖同名文件。

建议共享文件夹使用 `checkpoints/` 和 `runs/<alias>/<run-id>.zip`。小型、已审阅的 JSON/图表可以通过 PR 入 Git；checkpoint、数据缓存、大日志仍留在文件存储。其他成员取用时根据 run metadata 与快照核对，避免把不同代码/seed 的结果混成一次实验。

## 官方说明与当前边界

- [GitHub：邀请 collaborators](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository)：邀请需要对方接受。
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)：notebook 分享范围、私有 runtime、临时 VM 与账号权益。

启动器、恢复与导出流程先进行本地验证；这不等同于所有组员的 GitHub/Google 授权、GPU 分配和网络环境均已验证。首次 remote push 仍取决于本机 Git 授权，checkpoint 的共享云位置也尚未设定。
