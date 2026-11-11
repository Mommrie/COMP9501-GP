# 第二轮强化：loss权重、保守路由与更大评价区

本轮在读过v3与第一轮extension最终成绩之后设计，全部是探索性研究。旧结果保留；新计划与源码在运行前保存。复用原12个teacher、同三seed、同fit/selection，不增加独立训练重复数。

## 1. 为什么检查hidden权重

第一轮只测试固定λ=1，不能据此宣称所有hidden目标无效。本轮固定λ∈{0,0.01,0.1,1}；0和1端点复用同L4、FP32、同sample order的已完成结果，新增0.01/0.1的真实与shuffled配对目标，共12次新的系数拟合。每次仍是97参数、120步、相同学习率/正则与checkpoint候选。

λ及每条轨迹的step都只根据coefficient_selection的native score选择；相同分数选较小λ。λ=0就是CE控制，不是新增功能方法。真实和shuffled使用相同λ网格；它们比单CE拥有更多selection候选，因此任何提升都应解释为整个调参pipeline的结果，不能直接归因于独特activation信号。

## 2. 为什么保守路由需要回退

原expert router预测最小NLL专家ID，但这与“修复本任务模型错误，同时避免误伤”不是同一个目标。本轮提供：

- confidence gate：冻结已有expert分类器，只有最大expert概率过阈值才切换，否则用own expert；阈值0/.5/.7/.9及always-own。
- benefit gate：在fit上以各expert正确性减own正确性（−1/0/+1）为四维目标，用Ridge从标准化base feature与已知task one-hot预测相对收益。α=1/100/10000，切换阈值0/.02/.05/.1，另加always-own。

候选在selection上须满足native及mean accuracy不低于task-ID、worst accuracy最多低0.01，然后依次最大化native、mean accuracy、最少切换，剩余平局保留baseline。这只是selection约束，**不是final安全保证**。两种方法的参数量、fit标签和候选预算与静态方法不同。

全部选择写入JSON，再读取缓存internal final并载入官方validation。保存scaler与系数可重新加载gate，并检查重载与原模型路由一致。若选择always-own，可直接绕过base特征提取；这等于选择已有参照，不算新算法优势。其他gate仍需要额外特征前向，离线expert缓存不当作部署耗时。

## 3. 官方validation如何使用

在固定GLUE commit下使用全部validation。先按NFKC/casefold/空白归一化完整文本tuple，去掉与四任务**所有原train内容**相同的例子，再在每任务validation内去重；过滤完全不看标签。保留实际ID、content hash、fingerprint与剔除计数，样本量以实际manifest为准。

评价：task-ID/individual、原source/expert router、两种新gate、GT oracle、FP32 CE、selection-selected真实/shuffled目标，以及原v3 selection锁定的Average/TA/TIES/DARE。官方validation不参与λ、Ridge或阈值选择。它此前曾被旧研究探索，不能称新盲测；exact tuple过滤也不解决全部近重复和预训练污染。它是对更大、单独split的检验，**不是全新OOD证明**。

官方任务样本数不同，mean accuracy是四任务macro均值；native仍为SST/RTE acc、MRPC(acc+F1)/2、CoLA MCC。三seed误差条是sample SD。原A100毫秒数不能和本轮L4质量拼成实测Pareto改善。

## 4. 执行入口

[系数计划](../configs/activation-sensitivity-v2.json)、[路由/评价计划](../configs/routing-safety-v2.json)、[系数执行器](../code/activation_extension.py)、[选择与复评](../code/robustness_controls.py)。旧配置的hidden_weight默认行为保持相同；缓存仅在新配置显式开启时保存。

最终解释必须同时回答：较小λ是否改善、选择是否退回CE、gate帮助是否多于误伤、是否退回task-ID、较大split是否保留结论。任何结果都如实报告，不以预期胜率决定是否收录。

在Colab仓库根目录、已有v3及第一轮extension后执行（目录按实际路径替换，输出必须新建）：

```bash
python code/activation_extension.py --study outputs/my-v3 --plan configs/activation-sensitivity-v2.json --output outputs/my-sensitivity
python code/robustness_controls.py --study outputs/my-v3 --previous outputs/my-extension --sensitivity outputs/my-sensitivity --plan configs/routing-safety-v2.json --output outputs/my-robustness
python code/summarize_robustness.py --previous outputs/my-extension --sensitivity outputs/my-sensitivity --controls outputs/my-robustness --output artifacts/my-robustness
```
