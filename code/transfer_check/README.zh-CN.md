# 固定外部迁移检查

[原有说明](README.md) · [中文首页](../../README.zh-CN.md) · [扩展开发](../../docs/EXTENDING.zh-CN.md)

将 SST2 头用于 IMDb test，将 MRPC 头用于 PAWS labeled_final test；五种配置和系数沿用 source 上的既有选择，不在新目标上调参。每个目标 label-blind 抽 512 条，记录精确排重、版本、ID 和截断。

```bash
python code/transfer_check/run.py
```

需要三个标准 bank、coefficient-ce-control 的 selection，以及 selection-control 的锁定 KL 选择。脚本输出路径固定为 outputs/transfer-check，默认 runner 不提供新路径 CLI。summarize.py 读取归档 artifacts 文件名，而不是任意新输出；重跑后需显式整理新证据。

## 新目标怎么接

先在新 plan 冻结来源、split、字段、标签极性、采样与排重，再扩展数据适配。脚本当前仍有二类标签断言、现有头部和逐层索引假设；新类别或任务不能只换 repo 字符串。严格复现旧目标版本需显式支持并使用归档 revision，默认代码会重新解析当次 Hub commit。

保留当前负结果：PAWS 迁移连 individual 都低于子集多数类；IMDb 大量文本被长度128截断。不要看结果后改子集/长度还声称是同一次预定盲测。
