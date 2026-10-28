# 一起改项目的小建议

[中文首页](README.zh-CN.md) · [English](CONTRIBUTING.md)

大家用自己的账号提交即可。改动较大时可以开分支/PR 方便讨论，小修改按组内习惯来。

改实验时，写清运行命令、配置和结果就很有帮助。新数据或新方法可以参考 [扩展指南](docs/EXTENDING.zh-CN.md)；[实验记录模板](docs/EXPERIMENT_TEMPLATE.zh-CN.md) 只供按需取用。

修改 runtime 代码或配置后，记得更新 Colab 快照：

```bash
python scripts/build_team_notebook.py
```

需要时可运行 `python scripts/check_repository.py` 查语法、链接和快照是否过期。没有规定每次小修改都要跑完整模型实验。已有中英文指南可以继续补充，不作为新增模块的格式门槛。

密钥、较大的数据和模型权重单独保存；不同实验尽量用不同输出名，便于之后比较。保留真实贡献、引用和课程明确要求的信息。
