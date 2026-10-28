# Selection-label control

`run.py` 复用已训练的 KL 系数历史，在同一 selection 半集上比较 teacher-KL 选 step / uniform gain 与已有 GT 主指标选择。训练不重跑，选择候选固定为原来的 step 0/30/60/90/120，gain 1/2。无标签前向显式丢弃 labels，不计算 accuracy；最小化四任务平均 `KL(teacher || student)`，T=2。

这里“无额外 selection labels”仅指该新选择准则。原 task adapter/head 用过标签、teacher checkpoint 用过原 calibration 标签；selection 半集并非对 teacher 未见，整个研究已看过 validation，是探索性控制，不是完全无监督/无泄漏结果。验证分数不能用于改选候选。
