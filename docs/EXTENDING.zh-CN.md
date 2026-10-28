# 扩展开发指南

[中文首页](../README.zh-CN.md) · [文档导航](README.zh-CN.md) · [实验模板](EXPERIMENT_TEMPLATE.zh-CN.md)

本指南以当前代码为准。现在是已经跑通的研究工程，尚没有“注册一个插件即可自动支持任意数据集/模型”的接口。以下列出实际修改点与验收条件，避免队友只改配置后得到不兼容或信息泄漏的结果。

## 1 先选择扩展范围

| 想做什么 | 起点 | 需要一起检查 |
|---|---|---|
| 改学习率、样本数、batch、训练步数 | 新建模型配置，使用 train_lora.py 的 CLI | 独立输出、有效 batch、calibration 选择、输入协议 |
| 加合并方法或候选参数 | merge.py、merge_study/run.py 与新 plan | ΔW 维度、分类头、候选数、尺度与随机 mask 对照 |
| 加响应信号或层权重 | probe.py、functional_merge、coefficient_merge | 相同输入/掩码、冻结参数、层索引、优化容量 |
| 加 router | router_v1/run.py | source ID 与任务标签的权限、头部已知程度、额外前向成本 |
| 加外部评估数据 | transfer_check/run.py 与独立新 plan | label 极性、排重、截断、目标数据不调参 |
| 加训练任务、改类别数或 backbone | common.py 与全部受影响模块 | 下表中的固定假设必须先处理 |

## 2 当前固定假设：不能只改 JSON

| 代码位置 | 当前行为 | 扩展时要做什么 |
|---|---|---|
| common.FIELDS、selected_raw() | 四个已知任务；数据来源固定 nyu-mll/glue | 增加字段、来源、split 和标签规范；保留 sample ID/fingerprint |
| common.model_for()、metrics() | num_labels=2；F1 使用二分类默认口径 | 多分类同时改头部维数、F1 averaging 与标签校验 |
| merge_study.engine.task_score() | CoLA 用 MCC、MRPC 用 (acc+F1)/2，其余 accuracy | 为新任务明确主指标与汇总意义 |
| functional_merge/run.py | Gram、uniform、部分循环与汇总写死 4 | 推导为任务数 T，连同目标向量、归一化和测试一起改 |
| coefficient_merge/core.py | logits[...,4]、range(4)、uniform=.25 | 将 T 参数化；MixedLinear 的 factor 堆叠也要支持 T |
| coefficient_merge/run.py | 多处四任务平均；profile 层数来自 plan | 不漏改分母、初值、heads 和 LoRA factor 构造 |
| router_v1/run.py | 四个原型、二分类 logits、seeds 42/43/44 与 bank 路径 | 一起改任务数、类别数、索引、seed 和路径接口 |
| 各逐层合并脚本 | 从参数名 split('.')[2] 读取层号 | 换模型时用实际模块结构映射，勿假定名称兼容 |
| coefficient-v1.json | layer profile 为 24 层 | 与新 backbone 实际层数对齐，不只改 model 字段 |
| selection_control、transfer_check 的 runner | 固定已有输出路径与 history 名称 | 若要多组实验并存，先增加 CLI/config 路径参数 |

任务顺序也是协议：保存的权重矩阵第 i 列对应当时 cfg.tasks[i]。新增或重排任务后，旧系数不可直接复用。材料化 ΔW 合并、低秩 factor 混合对 rank/形状的要求不同；目前整条已验证流程统一 rank=8、相同 Qwen base。

## 3 最小扩展示例：学习率消融

在一个新分支中，从 protocol-v2-a100.json 创建新配置；不要改归档文件。以下 Python 只生成一个尚不存在的新文件：

```python
import json
from pathlib import Path

cfg = json.loads(Path('configs/protocol-v2-a100.json').read_text(encoding='utf-8'))
cfg['learning_rate'] = 1e-4
with Path('configs/my-lr-v1.json').open('x', encoding='utf-8') as f:
    json.dump(cfg, f, indent=2)
```

先用小预算检查新配置能运行，再扩展。下面命令会实际训练，需要已安装依赖和可用资源：

```bash
python code/train_lora.py --config configs/my-lr-v1.json --task sst2 --seed 42 --max-steps 3 --output outputs/my-lr-v1-smoke-s42
```

验收：输出包含配置/环境、数据 ID 和保存的 adapter/head，loss 与参数更新可检查；3 步成功只是连通性检查，不是性能结论。之后按预先固定的 seeds、预算和 calibration 规则跑完整实验。修改模型配置不保证所有 run_all.py 自动采用它，调用前检查其固定路径和参数。

## 4 新增训练任务或数据源

1. 在新 plan 写清数据发布方、revision、字段、类别数/极性、split、采样 ID、许可与评价口径。先检查数据和任务定义是否与现有分类头一致。
2. 扩展 common.py 的来源加载与字段映射。保留 train/calibration/validation 的分离；多分类与非 GLUE 数据还要改分层切分、manifest、head 和 metrics。
3. 先验证单样本和长句对：Sentence A/B 顺序、有效 token、截断、label mapping、padding、pooling。新任务提示不能偷偷暴露真值标签。
4. 单任务小样本拟合与 head-only 对照通过后，重建该协议的 adapter bank。输入格式、base revision 或头部定义变化时，旧银行只作为旧协议记录。
5. 按固定假设表处理所有 T=4、两类和 task order 的地方，再做跨任务合并。不能把五个任务的分数除以 4。

模型 commit 已在默认配置中固定；默认 GLUE 配置未给出 dataset_revision。已有 fingerprint/ID 记录不等同于完整数据版本锁定，严格重现需要补足确切数据 revision。现有 transfer runner 会在加载前解析并记录目标 Hub commit；重跑默认脚本不保证自动回到旧目标 commit。

## 5 新增合并算法

算法核心从 merge.py 的 merge_tensors() 入手，输入是同一参数位置上的材料化 ΔW，输出形状保持一致。LoRA 的 ΔW=(alpha/r)BA，不能分别平均 A、B 后当作平均 ΔW。

接入要同时覆盖：算法实现、CLI 的 method choices/必要参数、engine.merge() 的候选传递，以及新 plan 的候选/选择规则。目前 engine 传递的是 method、scale、density、seed；若新方法需要 activation、独立权重或额外统计量，应显式扩展接口，不能只往 JSON 加一个未读取的键。

至少核对以下有实际意义的性质：

- 输出与输入目标层的形状/键一致，全部数值有限；非目标参数不变。
- 统一加权退化到 average、one-hot 权重退化到单个 adapter 等可推导边界。
- 反复切换候选时，状态始终由 immutable base + delta 构建，没有残差漂移。
- 使用原任务头时能复现 individual；若改头部，另立信息权限和比较协议。
- 随机 mask、缩放、搜索次数与 baseline 匹配。别把更大调参预算当成新信号效果。

## 6 扩展 activation / 系数学习

probe.py 在共同 calibration 输入上记录最后层 mask-aware mean，delta 是 adapter−base。换层、换 pooling 或来源输入时同时记录定义，保证逐样本对齐。raw cosine 高可能主要来自共同 base，不等于功能增益。

functional_merge 使用 task-conditioned q/v 线性响应 Gram 与 simplex 约束。共同输入的平方响应匹配有 uniform 最优解，保留这个解析控制。局部 surrogate 下降不能直接推出整体分类损失下降。

coefficient_merge 只训练系数和 gain。新增机制需要检查 frozen base/LoRA/head 的梯度、初始 logits 等价性和最终 materialized delta 评估；不能把训练时执行多个 expert 的成本当成部署时单模型成本。同容量 CE、uniform/gain 和初始 step=0 都应作为对照。

Teacher-KL 不用 fit 标签，不代表全流程无标签：原 teacher/head 训练、teacher checkpoint 选择及部分选择规则使用过标签，当前 calibration 半区对 teacher 并非全新。

## 7 扩展 router 或 OOD 检查

当前 router 学的是数据源 task ID，任务分类头仍由已知 task ID 提供。source accuracy、逐样本最优 expert 和真实部署任务不等价。新增 router 先写清输入、可用标签/任务元数据、头部来源与前向次数，再考虑未知任务、style shift 或其他数据域。

新增 target 数据时，在加载/看结果前冻结选择规则和 label mapping；记录精确去重、样本 ID、Hub commit 与截断比例。目标标签不能再次用于选择 source 的 gain、step 或最好看的数据子集。当前 IMDb/PAWS 负结果保留，不能通过事后改长度后仍称原方案首次盲测。

## 8 验证与提交

新模块可以补一小段用途和运行方式；中英文说明按组内需要维护。只有受影响逻辑需要新增相应数值测试；文档修改运行链接/结构检查即可。按实际修改范围运行：

```bash
python scripts/build_team_notebook.py
python scripts/check_repository.py
python -m unittest discover -s code -p test_math.py
python -m unittest discover -s code -p test_protocol.py
```

系数/functional/CSA/打包工具另有各自测试，见各板块中文指南。随机微型 Qwen 集成检查需要固定 transformers/peft 环境，但不等于新 GPU 实验通过。

较大的实验可以附上配置、版本、数据和 seed 等背景，方便组员接着跑。模板见 [实验与 PR 清单](EXPERIMENT_TEMPLATE.zh-CN.md)。保留原实验结果，不把新运行覆盖到 2026-10-25 记录中。
