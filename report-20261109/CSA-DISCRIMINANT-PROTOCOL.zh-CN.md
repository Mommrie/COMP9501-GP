# CSA判别学习与task-ID残差路由：冻结协议

2026-11-11。[配置](../configs/csa-discriminant-v1.json) 与 [源码](../code/csa_discriminant.py) 在计算前冻结。复用先前CSA/PCA投影、全部响应及旧fit/selection/final角色，不重新训练teacher、不重拟合CSA basis。此前final已看过，因此是探索性方法迭代。

## 理论联系与实现

选文部分识别实验将CSA与LDA组合。这里沿用“先结构化降维，再判别学习”的方式：CSA仍只最小化无标签重建误差，后接共享类内协方差的正则化线性判别分析；不把它称为新提出的监督CSA。原文为对象类别识别，我们改为expert/切换效用类别，监督目标和数据对象已改变。

LDA协方差为 `(1−α)Sw + α·trace(Sw)/D·I`，α固定为.1/.5/.9；经验类先验，scaler只在fit学习。高维时用Woodbury恒等式计算相同线性判别函数，避免raw_delta的巨大特征协方差求逆；数学结果在低维/高维随机例上与scikit-learn `solver=lsqr, shrinkage=α`核验。若类内方差退化，使用1e−8数值下限。只有一个类别时返回该类概率1。

## 三种方法，所有表示使用相同网格

1. **Direct LDA**：fit目标为最小真实标签NLL的expert，直接选LDA后验最大者。只按selection native选α，平分取较大α。
2. **Task prior**：仍预测上述expert后验，但默认own expert。只有最优其他expert的概率超过own概率一定margin才切换。margin=0/.05/.1/.2/.4/.7，外加always-own；概率相同保留own。
3. **Benefit LDA**：每个候选expert分别学习help(+1)/tie(0)/harm(−1)，来自该expert正确与own正确之差。对应expert自己的任务行必然差为0，因此不用于它的benefit分类器拟合（每个384条）；类别不平衡和单类情况保留。以P(help)−P(harm)估计收益，只有超过0/.02/.05/.1/.2的阈值才切换，外加always-own。

表示覆盖原七个CSA rank、全部同latent/同预算PCA rank、raw_delta及base。先标准化表示，再拼接已知task one-hot；原logistic结果保留作为历史matched-data参照。新的判别/收益目标不是原logistic的单因素比较，不能把整体差值全归于LDA。

## 选择与评估

- scaler及LDA用512条coefficient_fit，alpha/margin/threshold用512条selection。
- Task prior/benefit gate要求selection native与宏平均accuracy均不低于task-ID，worst accuracy最多低.01。合格者先最大native，再最大宏accuracy，再最少切换；剩余平分偏向always-own、较强收缩和较大阈值。包含always-own确保有参照候选，但不保证final安全。
- 每seed、每表示、每方法单独锁定；全部锁定后才读取1,024条内部final的标签/logits/响应。所有rank报告，不按final挑冠军。
- 主要展示固定的CSA2×8、同latent PCA16、预算PCA3；raw/base做匹配参照。其他rank是完整备份，不从它们事后挑最好者替代主结果。
- 必须报告native、平均/worst accuracy、switch fraction、help/harm、三seed逐项结果及always-own次数。`help−harm = 宏accuracy−task-ID宏accuracy`，不等于native分数差。native包含MRPC F1与CoLA MCC，不可当准确率百分比。

## 不提前承诺的结果

接近task-ID必须区分：若三seed全部always-own，只说明保住基线；若有非零切换，需要看final净收益、最差任务及seed一致性，不能因selection合格就称安全。若LDA反而更差，也保留。每输入依旧需要全部expert响应，本轮不建立推理成本优势。

## 复现

```bash
python -m unittest discover -s code -p test_csa_discriminant.py
python code/csa_discriminant.py --bridge outputs/csa-bridge-20261111 \
  --previous outputs/activation-sensitivity-20261111 \
  --plan configs/csa-discriminant-v1.json --output outputs/new-csa-discriminant
```

需要匹配的CSA state备份、原三个角色的cache及投影。计算在Colab CPU上复用缓存，本地只编辑、传输与核验；GPU无需新增模型任务。
