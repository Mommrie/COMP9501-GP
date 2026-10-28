# 文档与扩展导航

[原有说明](README.md) · [中文首页](../README.zh-CN.md)

如果只想跑现有工程，先看根目录中文 BUILD；如果要改代码，先看 EXTENDING，再按实验模板写计划。每个主要目录都有 README.zh-CN.md，通过中文首页的表格进入。

## 文档地图

- [Proposal 与进度](../PROPOSAL.md)：原始英文提案与中文整理、现有结果对应关系、尾部 timeline 和完成/待办清单。
- [扩展开发指南](EXTENDING.zh-CN.md)：真实修改点、固定四任务/二类/层数假设、配置消融例子、新数据/算法/模型接入与验收。
- [实验与 PR 模板](EXPERIMENT_TEMPLATE.zh-CN.md)：信息权限、预算、版本、ID、seeds、结果与提交清单。
- [中文版运行说明](../BUILD.zh-CN.md)：环境、bank、各后续模块的依赖顺序与排错。
- [中文版组员开发流程](../CONTRIBUTING.zh-CN.md)：身份、分支、同步、检查与 PR。
- [Colab 协作说明](../COLLABORATION.md)：私库授权、独立 GPU、恢复和结果保存。

## 维护这些指南

扩展代码时核对文档的 CLI 与实际 argparse，明确命令需在哪个目录运行、输出是否必须新建、依赖哪些已有 bank/history。新增模块时可以补上简短用法，避免只写“参见其他目录”而缺少具体用法。

不要把未来准备支持的接口写成已经实现。源码/配置变化后重建两个 launcher；中文说明变动用检查器验证链接和覆盖，保持中英 notebook 的可执行 cell 一致。
