# 小工具

[中文首页](../README.zh-CN.md)

- `check_repository.py`：需要时查语法、JSON、链接、明显的密钥/本地路径和快照状态；不规定 PR、文档语言或提交格式。
- `build_team_notebook.py`：代码/配置变动后更新中英 notebook，`--check` 只检查；英文说明变动时同步 JSON 翻译。
- `colab_bundle.py`：恢复 checkpoint 与打包结果，保留文件校验和避免意外覆盖的保护。
- `test_colab_bundle.py`：打包工具的离线检查，改相关代码时可运行。

```bash
python scripts/check_repository.py
python scripts/build_team_notebook.py
```

这些工具可以按需要使用，小文档改动通常检查链接即可。
