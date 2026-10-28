# 归档结果与证据

[原有说明](README.md) · [中文首页](../README.zh-CN.md) · [扩展开发](../docs/EXTENDING.zh-CN.md)

这里保存已审阅的数值 JSON、生成图表、冻结方案和校验记录；outputs/ 的新运行不自动成为本目录的归档结论。

## 常用入口

- replication-summary：三训练 seed 的基线与任务对汇总。
- prototype-aggregate：局部响应匹配、CE/KL 系数与 router 的逐项汇总。
- selection-control-aggregate：GT 与 teacher-KL 选择的对照。
- transfer-check-aggregate：IMDb/PAWS 固定迁移检查。
- source-provenance：初始导入文件哈希及共享适配说明。
- team-launcher / team-workflow-validation：代码快照与本地恢复/导出验证，不能代替 live Colab GPU 验证。
- chinese-guides-validation：中文目录覆盖、链接、双语 notebook 代码一致性及源码未变校验；不是新模型实验。

报告链接在中文首页；各 seed 原始 JSON 保留，用于核对 mean±sample SD。备份 manifest 中的私有 ZIP 不随 Git 分发；不要把清单当成已下载的权重。

## 怎样添加新结果

使用新日期/协议名，附 cfg/plan、code commit 或快照 SHA、数据 ID/revision、每任务每 seed 数字、选择规则与资源记录。说明 SD 的统计单位，保留负结果和失败 seed；六个共享任务的 pair 不能当作独立样本。

三 seed 和已查看过的 validation 仍是探索性证据。论文式叙述要能回到对应 JSON，不用更好看的平均值覆盖旧结果。图表与文本变动后检查真实单位、图例、误差棒和主指标定义。
