# Feature-only hard routing prototype

Router fit/selection 使用 calibration 前/后半部分，目标为数据源 task ID，不用分类标签。原 task head 在评估时继续由 task ID 提供，因此不是未知任务头的统一分类器。

比较 random、由真实输入字段数量决定的 format-only、base feature prototype、calibration 选 C 的线性路由与 task-ID 直接选择。最后一种是当前已知任务头协议下的重要参照，不应隐藏；不称为标签逐样本最优的上界。

Base mask-aware mean 特征要额外一次 backbone 前向；prototype/linear 路由加上所选 adapter 合计两次。Random / format / task-ID 只需 adapter 一次。硬路由可共享 base 和 adapter 存储，不能用理想 FLOPs 冒充实际 wall-clock 加速。
