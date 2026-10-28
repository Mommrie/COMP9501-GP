# End-to-end static coefficient fitting

这是本轮探索的第二个静态原型：只训练 adapter 的混合系数和一个公共 gain，base、LoRA A/B、分类头全部冻结。比较使用 calibration 标签的 CE 与匹配 individual teacher logits 的 KL；不声称系数学习或蒸馏本身是新方法。

原 calibration 前 128 条/任务用于拟合，剩余至 256 条用于选择 step（RTE 121 条）。候选 step 包含 0，模型选择只用后半部分主指标均值；锁定后再评估 validation。所有 validation 已被前期探索使用，结果是课程项目的探索性证据，不冒充新的独立确认集。

拟合时以低秩响应实现精确的加权 ΔW，最终 materialize 为单个 backbone。`core.py` 的线性等价性与冻结参数单元检查通过后才能运行。配置见 [coefficient-v1.json](../../configs/coefficient-v1.json)。
