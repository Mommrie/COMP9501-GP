# 构建与检查工具

[原有说明](README.md) · [中文首页](../README.zh-CN.md) · [扩展开发](../docs/EXTENDING.zh-CN.md)

这些工具管理共享工程和 Colab 执行副本，不改变科学训练算法。

| 工具 | 用法与结果 |
|---|---|
| check_repository.py | 标准库检查 Python/JSON/notebook、相对链接、凭据/本地路径、中文指南与 launcher 同步 |
| build_team_notebook.py | 生成中英 team notebook 及快照 manifest；--check 只验证不写文件 |
| colab_bundle.py | 校验源包、恢复 checkpoint、保护已有内容、导出带文件 SHA 的 run |
| test_colab_bundle.py | 检查幂等、冲突、损坏、路径穿越和导出 metadata |

从仓库根目录执行：

```bash
python scripts/build_team_notebook.py
python scripts/build_team_notebook.py --check
python scripts/check_repository.py
python -m unittest discover -s scripts -p test_colab_bundle.py
```

## 给后续开发者

source-provenance 保存初始导入事实，合法后续改动会列入 changed_since_archive_import。默认不因修改过代码而拒绝提交，--strict-import 才要求原样。不要重写旧 manifest 来掩盖协议变化。

扩展打包范围时修改 bundle_files() 的明确文件选择，并核对两个 notebook 的快照、依赖、恢复与导出。恢复流程按清单与 SHA 验证全部成员后写入，不使用无校验覆盖；新增文件类型时保留这些保证。修改 notebook 说明或翻译后也使用生成器，不手改生成文件。

英文 notebook 说明变化时，用 `--print-markdown-digest` 查看新哈希；审阅对应中文文字后更新 notebooks/team_notebook.zh-CN.json，再生成。该选项只输出哈希，不修改文件。
