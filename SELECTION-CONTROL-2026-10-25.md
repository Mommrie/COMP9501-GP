# Selection 标签控制：复用既有 KL 系数轨迹

2026-10-25。固定已有 global/layer KL 的 step 0/30/60/90/120 与 uniform gain 1/2，改为在同一 selection 半集最小化四任务平均 `KL(teacher || student)`（T=2）。没有新增训练；选择阶段前向丢弃 class labels，不计算 accuracy。先保存锁定候选，再评估完整 validation。

三 adapter 训练 seeds 42/43/44，四任务主指标定义与 [原型报告](PROTOTYPES-2026-10-25.md) 相同，± 为 sample SD：

| 方法 | 原 GT 主指标选择 | 本次 teacher-KL 选择 |
|---|---:|---:|
| uniform | 0.6365±0.0365 | 0.6365±0.0365 |
| global KL | 0.6479±0.0119 | 0.6497±0.0041 |
| layer KL | 0.6955±0.0053 | 0.6996±0.0126 |

Uniform 三次仍选 gain 2。Layer KL 三次均选 step 120，global KL 分别选 120/90/120。Layer KL 的 validation 分数为 0.712811、0.687781、0.698070；相对 GT 选择的变化并非三个 seed 都为正。这一控制表明，在这些固定候选上，**不向选择准则增加真实标签仍能保留改善**；不能据此证明 teacher 目标优于同容量 CE（其原 GT 选择均值为 0.6997）。

重要边界：teacher/head 的训练已用标签，原 teacher checkpoint 也已用整个 calibration 选择；selection 半集只对本次系数 fit 分开，并非对 teacher 全新。前期研究已查看过 validation，本次是探索性后续控制。不能称为完全无监督、独立盲测或首创新算法；也未通过这一控制解决跨数据集泛化和输入风格混杂。

执行耗时约 202 秒；三 seed 全部通过，JSON、锁定候选、执行源码和配置已本地备份并核验 SHA/CRC/源码一致性。原系数历史、原 GT 结果与代码保持原状。

- [原始结果](artifacts/selection-control-2026-10-25.json)、[汇总](artifacts/selection-control-aggregate-2026-10-25.json)、备份校验（本地记录）。
- [代码与信息边界](code/selection_control/README.md)、[固定候选方案](configs/selection-control.json)。
