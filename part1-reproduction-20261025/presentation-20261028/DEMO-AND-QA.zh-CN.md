# 代码演示路线与问答

入口：[材料包](README.md) · [公式对应](CODE-WALKTHROUGH.zh-CN.md) · [图表目录](FIGURE-GUIDE.zh-CN.md)

这份材料供负责实现的人准备讲解，并不要求接手整份 PPT。课程 15 分钟 presentation & Q&A 中的代码部分应与组员协调；以下 3 分钟路线可独立缩短为一张图和一个代码片段。

## 约 3 分钟代码路线

**0:00–0:35：输入与输出。** 展示 01 和 02：`(N,14,14)` 输入，训练均值中心化，`(N,4,4)` core，重建回原图。明确 fit 不读标签，分类是后接的 1-NN。

英文讲述：*Each image remains a matrix. We learn two orthonormal projections from the training images, and each projected image is a small core. The reconstruction uses the same projections in reverse.*

**0:35–1:25：四行核心更新。** 打开旧 `csa.py` 的 `fit`：`zv`、U 更新、`uz`、V 更新。解释每次先压缩其他方向，再求当前方向的条件协方差主特征向量。新 U 立即供 V 使用。

英文讲述：*Once V is fixed, updating U is an ordinary leading-eigenspace problem. We then use the new U to update V. Each conditional step maximizes retained training energy, but the joint problem is still nonconvex.*

**1:25–2:00：为什么相信代码。** 展示 07 曲线和通过测试的摘要：正交、目标不减、能量恒等式、PCA 训练下界；1D 等于 PCA，2D 与旧实现一致，3D 能重建新的已知低秩张量。无需现场展示整屏测试输出。

英文讲述：*We check algebraic identities rather than judging correctness from classification accuracy alone. Monotonic energy and exact synthetic reconstruction are implementation checks, not a proof of universal superiority.*

**2:00–3:00：一个带条件的结果。** 用 06+04/05：相同 latent=16 时 PCA accuracy 更高，CSA basis 更小；存储上限相同条件下 PCA 只能保留 9 维，此时 CSA accuracy 更高。把预算说完整，避免只说“CSA 更好”。

英文讲述：*The comparison budget matters. At equal latent size, PCA is more flexible. With basis storage included, CSA can afford a larger core under the same storage limit. Our results illustrate that tradeoff rather than a universal ranking.*

## 现场运行建议

优先使用本地缓存和预先生成的图，网络下载不放在正式展示的关键路径上。CPU 足够，不需要现场开 GPU。

```bash
python -m unittest discover -s part1-reproduction-20261025/code -p "test_csa*.py" -v
python part1-reproduction-20261025/code/prepare_presentation.py --data /path/to/mnist.npz --output part1-reproduction-20261025/outputs/live-demo-unique-name
```

输出名每次不同，已有目录会拒绝替换。运行完成应看到 `PART1_PRESENTATION_PASSED`。若现场运行环境不稳定，直接用已验证图片和 CSV，说明它们是预先计算的真实结果即可。

## Q&A 备答

| 问题 | 回答要点 |
|---|---|
| 你们具体实现了什么？ | 二维 CSA/PCA 基线、通用张量 CSA、可复现的对照/绘图脚本和数学验证；不是原作者代码。 |
| 这算论文复现吗？ | 是基于原重建准则的机制实现与补充实验；原论文的 Gabor/人脸/LDA 表格未严格复现。 |
| 为什么不直接用现成库？ | 自实现能把每个条件协方差和投影步骤对应到 Theorem 1，也便于独立数值验证；不是主张现成库不可靠。 |
| “concurrent”是并行算的吗？ | 这里指共同作用的多个 mode 子空间；代码按 mode 顺序更新，不代表多线程同时更新。 |
| 向量化丢失信息了吗？ | 像素重新排列本身可逆；差别是学习的投影是否受可分离结构约束。 |
| 为什么不选识别率最大的 rank？ | 新图用于展示完整权衡，未设独立 selection 集；按 test 选 rank 会使报告乐观。 |
| 为什么 PCA 训练误差总更低？ | 同 latent 的 CSA 子空间是 PCA 可行域的子集；这是灵活性代价，不是实现失败。 |
| 那 CSA 为什么可能有用？ | basis 更紧凑，有限样本下的结构正则和存储约束可能有价值；效果需在明确预算和数据上检验。 |
| 已经证明更省存储了吗？ | 已计数 basis/core/mean 标量；文件格式、数值精度和压缩编码不同会改变实际字节，未测的部分不外推。 |
| 28 倍是整体压缩比吗？ | 不是。r=4 的 basis 为 3136/112=28；加 300 张 core 和均值，总数为 8132/5108，约 1.59。 |
| 为什么预算内 PCA 不是 16 维？ | 每个 PCA basis 方向需 196 个数，CSA 两个小 factor 更省，因此同存储上限下允许的 core 维数不同。 |
| rank1 的 10% PCA 是怎么回事？ | 上限容不下一个方向，d=0；mean-only core 距离全并列，固定预测一个类。它是退化点，不能当强对手。 |
| 为什么不用论文停止规则？ | 我们选择符号/基旋转不敏感的目标变化标准并明确记录差异；原文相似度不等号是 `>`，本次已看图核对。 |
| 迭代次数是不是越少越好？ | 依赖初始化、容差和每步成本；目标收敛与参数稳定不同，不能跨停止规则直接比较。 |
| 随机初始化会给不同结果吗？ | 可能；这次固定小子集的 9 次拟合结果几乎一致，不能推断所有情况。 |
| 每类只有一张图能做吗？ | CSA 可拟合，但类别标签不是降维的输入；“每类一张”通常仍有多个类别/样本，不等于中心化后仅一个训练样本。 |
| 3D 误差几乎为零说明能完美压缩图像吗？ | 仅对按已知低秩无噪声模型生成的数据；真实图像不严格满足该模型。 |
| 错误棒是什么意思？ | 三个训练子集结果的样本标准差；共享固定 test，不是置信区间或 3000 个独立 test 预测。 |
| 你们是否更改过旧结果？ | 旧源码/plan/结果保留，新目录和新 runner 保存扩展；六个重合设置与旧指标一致到数值误差。 |

## 向 PPT 制作者交付时附一句话

请保留每图的比较条件与来源：04 是相同 latent，05 是总存储上限，10 是原作者 Table IV；02/03 是固定模型的示例，09 是合成数据验证。正文可少用图，完整结果与代码解释放附录备用。
