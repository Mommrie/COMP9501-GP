# 组员开发与提交：中文版

[中文首页](README.zh-CN.md) · [English](CONTRIBUTING.md) · [扩展开发](docs/EXTENDING.zh-CN.md)

本维护者的提交署名为 `lrtang`。组员保留各自真实 Git 身份、贡献和引用；不把所有人的工作统一改成维护者姓名，不添加自动工具联名。课程要求的披露仍按实际规则处理。

## 开始一项改动

```bash
git fetch origin
git switch -c feat/your-alias-topic origin/feat/team-colab
```

当前团队功能与中文文档在 feat/team-colab；它合入 main 后，新开发可改从 origin/main 建分支。不要在有未保存改动时机械执行切分支命令，也不要强推覆盖共享历史。维护者的本地 user.name 不会自动配置到其他组员的 clone。

必要时只设置当前仓库的身份，填写自己的信息：

```bash
git config --local user.name "你的提交署名"
git config --local user.email "你的Git提交邮箱"
```

## 修改范围与文档

新增模块提供 README.md 和 README.zh-CN.md，分别说明用途、输入/输出、完整 CLI、默认路径、依赖、限制和扩展点。配置、代码和结果一起变化时，说明是哪一项研究假设，而不是只堆参数。

先复用 [实验与 PR 模板](docs/EXPERIMENT_TEMPLATE.zh-CN.md) 写清预算和信息权限，再跑目标评价。新 runtime 源码/配置变化后重建中英 notebook；文档翻译则确认链接和可执行 cell 一致即可。

```bash
python scripts/build_team_notebook.py
python scripts/check_repository.py
```

按受影响的逻辑运行模块数值测试。没运行的 GPU 或集成验证明确写“未运行”；本地打包成功不能替代 Colab GPU 验证。

## 提交与 PR

只 stage 本次改动，先审阅 diff，再用自己的身份提交、推送分支：

```bash
git diff
git status --short
git add docs/your-guide.md
git commit -m "Describe the concrete change"
git push -u origin HEAD
```

这里的文件名和分支名是示意，按实际文件替换。PR 写清问题、行为变化、验证结果和实验局限；当前以团队分支为基线时选相应 target，之后按 main 工作流集成。合并前同步最新代码，保留双方修改并处理冲突。不要为了让检查通过而删除原结果或改写其他组员历史。

权重、大日志、数据缓存留在授权文件存储；小型结果 JSON/图表审阅后可入 Git。不要提交 token、课程附件、个人资料或其他课程材料。
