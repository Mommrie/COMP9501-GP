# Fixed external-dataset transfer check

`run.py`：冻结已有 individual / average / uniform gain2 / layerCE-GT / layerKL-teacherKL 五种配置，分别用 SST2 头在 IMDb test、MRPC 头在 PAWS labeled_final test 检查迁移；每个新数据集固定 seed 950125、label-blind 抽 512 条。数据集公开版本在加载前解析为 Hub commit 并记录。新数据不选系数、gain、step 或任务头。

与已接触的对应 GLUE train+validation 做 lowercase/whitespace 规范化的文本精确排重（句对忽略次序）；不宣称去除了近重复或预训练污染。沿用长度128与原 sentence_fields_v2，报告截断比例。数据域、文本长度与任务定义均可能不同，不把这两项外部检查称为完整 GLUE 分数或全领域鲁棒性。

`summarize.py` 汇总三 seed 的 sample SD 与图；多数类基线按固定目标子集比例计算，不用于改选模型。
