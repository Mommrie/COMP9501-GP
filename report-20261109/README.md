# Part 2 最终报告、结果与展示材料

2026-11-11 已完成三seed云端复核、预测独立审计、11页英文LaTeX报告编译及逐页视觉检查。本目录提供给组员准备正式slides和presentation；行政场次、成员分工和课程正式提交另行核对。

建议阅读顺序：

1. [最终结果精讲与英文口答](ACTUAL-RESULTS.zh-CN.md)：主结果、负结果、配对差异、oracle和帮助/损害分析。
2. [英文报告PDF](report.pdf) 与 [LaTeX源码](report.tex)：摘要、相关工作、方法、实验、结果、成本/OOD、局限及引用。
3. [中文机制阅读指南](READING-GUIDE.zh-CN.md)：代码、proposal、paper和结果之间的联系。
4. [12页presentation内容与英文讲述](PRESENTATION-MATERIALS.zh-CN.md)：已填入真实结果，配图、时长与备份页建议；不是最终PPTX。
5. [15组强化机制Q&A](EXTENSION-QA.zh-CN.md) 与 [26组实现Q&A](../PART2-IMPLEMENTATION-QA.zh-CN.md)。

## 证据与代码

- [新L4完整证据](../artifacts/activation-strengthening-2026-11-11/README.md)：五个FP32 static controls、source/expert router、task-ID/random/oracle、完整JSON/CSV及小型派生预测npz。
- [冻结v3结果](../PART2-RESULTS-V3-2026-11-09.zh-CN.md)、[A100成本实测](../PART2-FOLLOWUP-2026-11-09.zh-CN.md)、[旧partial恢复说明](../artifacts/recovered-partial-2026-11-09/README.md)。恢复重跑使用同12个teacher/三个seed，不增加独立重复数。
- [proposal映射](EXPERIMENT-AND-CLAIM-MAP.zh-CN.md)、[相关工作边界](../PART2-RELATED-WORK.zh-CN.md)、[配置](../configs/activation-strengthening-v1.json) 与 [执行代码](../code/activation_extension.py)。
- [预测审计/图表生成器](build_materials.py) 与 [云端编译导出脚本](finalize_cloud.py)；原模型/GLUE版本和分区SHA均保留。模型权重和原始数据不放Git。
- [编译与渲染记录](build-verification.json)、[逐页视觉检查](visual-review.json)、[云端代码测试记录](code-validation.json)。六套15项开跑测试，另有快照跨平台回归、四项打包测试和一项微型训练/合并集成测试，共21项；双语notebook在Linux上同步校验通过。

## 使用和复现

仓库根目录执行 `python scripts/verify_part2_artifacts.py` 可用标准库核验三套证据清单及最终PDF；`python scripts/check_repository.py`检查语法、链接和源码快照。

模型复现请用 [中文团队Colab notebook](../notebooks/team_colab.zh-CN.ipynb) 的v3恢复/重新训练和activation controls开关。完整结果产生后，`build_materials.py --study ... --extension ... --artifacts artifacts/my-new-run` 可审计并生成表图，输出目录须不存在。脚本的v3主表和历史OOD表依赖原研究证据；新报告重编译时同步TeX中的图路径，不覆盖冻结记录。

重编译已归档报告：在本目录运行 `tectonic report.tex`，或使用带标准LaTeX包的兼容工具链。`generated-results.tex`及四份`table-*.tex`是审计脚本生成的数据表，不手工更改数值。图像通过相对路径引用仓库artifacts，因此请保留目录结构。

本轮结论支持学习小型静态混合改善当前固定基线；未建立输出KL/hidden信号超出CE的独特收益，expert router也未优于已知task-ID。所有扩展仍是对已探索内部final的探索性研究，不称外部盲测或SOTA。
