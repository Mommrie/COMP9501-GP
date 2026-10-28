# 新实验与 PR 模板

[中文首页](../README.zh-CN.md) · [扩展指南](EXTENDING.zh-CN.md)

复制下列内容到新实验计划或 PR 描述，按实际情况填写。尚未完成的验证明确写“未运行”，不要把计划写成结果。

## 问题与变化

- 要检验的假设：
- 相对哪个已有 baseline，具体改变什么：
- 预先确定的失败判据或反例：
- 代码分支/commit 与配置/plan 文件：
- 是否改变任务、头部、输入模板、标签数、base revision 或数据版本：

## 信息与预算

| 项目 | 本次约定 |
|---|---|
| train / calibration / validation / target 数据 | |
| fit、checkpoint 选择、最终评价分别读取哪些 labels/teacher/元数据 | |
| sample IDs、fingerprint、revision、精确排重方式 | |
| adapter seeds、采样 seed、随机 merge mask seed | |
| epochs/steps、batch/累积、样本数、precision | |
| 候选数、gain/density/temperature 等选择范围与 tie-break | |
| 比较的参数容量、搜索预算、前向次数/实际运行时间 | |

## 复现命令

写出从仓库根目录执行的完整命令、新输出目录和依赖。依赖已有 bank/历史文件时标明名称、SHA 与来源；不要提交权重、token 或本地绝对路径。

## 验证与结果

- 实际运行的检查及结果：
- 未运行/无法验证的部分：
- 每个任务、每个 seed 的主指标及必要辅助指标：
- mean±sample SD 的统计单位与 n：
- 资源测量、负结果、失败 seed、塌缩或边界情形：
- 与假设的关系、尚不能推出的结论：

## PR 提交清单

- [ ] 新文件有 README 导航与中文说明；CLI/输入/输出路径已核对。
- [ ] cfg 与 plan 分工明确；新键确实被 runner 读取；四任务/二分类/层数假设已检查。
- [ ] 结果保存在新目录；原 evidence、真实贡献与引用保留。
- [ ] 新 runtime 源码/配置已重建中英 Colab launcher，快照 SHA 对应当前内容。
- [ ] 仓库检查通过，受影响的数值/集成检查已运行或解释未运行原因。
- [ ] 未使用目标评价数据反复调参后仍声称盲测；旧 validation 探索边界已披露。
- [ ] 提交使用本人身份，无自动工具联名；没有覆盖或重写其他组员的历史。
- [ ] 只向约定的 private repo/文件存储共享；课程附件、原数据与敏感材料未混入。
