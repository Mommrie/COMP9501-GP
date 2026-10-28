# 共享权重与恢复

[原有说明](README.md) · [中文首页](../README.zh-CN.md) · [扩展开发](../docs/EXTENDING.zh-CN.md)

[checkpoints-2026-10-25.json](checkpoints-2026-10-25.json) 是已有 12 个任务 adapter、bank 元数据和 coefficient/selection histories 的清单。约 24.6 MB 的 ZIP 在本地，不进 Git；共享云端下载位置尚未配置。

## 组员怎样用

组内将清单对应的 ZIP 放在约定的 private 文件存储并单独授权。组员下载后上传到 Colab Files，或给出自己的已授权 Drive mount 路径，填入中文 notebook 的 CHECKPOINT_ARCHIVE。恢复器验证整个 ZIP 和逐文件 SHA，已有内容不同时拒绝覆盖；恢复后 notebook 可直接复评单个 adapter。

Base model 和公开数据仍从原发布方下载，包内不含原数据、base 权重、其他课程材料或凭据。GitHub 邀请不自动授予 Drive 权限。

## 新增资产

用新文件名和新 manifest 记录 bytes、SHA、文件列表、model/协议/seed 与来源。保留旧版本，确认 adapter rank/目标层、分类头、bank 路径与当前 runner 兼容后，再更新恢复入口。清单不是 checkpoint 本身，只有 JSON 不能重新做模型推理。
