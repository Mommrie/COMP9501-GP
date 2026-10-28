# Part 1 CSA 补充复现

[原有说明](README.md) · [中文首页](../README.zh-CN.md) · [扩展开发](../docs/EXTENDING.zh-CN.md)

这里用二维灰度 MNIST 演示 CSA 的条件特征分解、重建与识别。它不是原论文人脸/Gabor-MNIST 表格的严格复现。先读 RESULTS.md，再用 PRESENTATION-OUTLINE.md 准备讲解和问答。

**新增：给代码讲解人与 PPT 制作者的 [完整材料包](presentation-20261028/README.md)**。包含公式→代码说明、原文证据、10 组 PNG/SVG、同 latent / 存储预算两类实验、初始化与三维验证、讲解路线与 Q&A。旧结果保留；新结果完整报告，不作普遍优势或严格复现原表的声明。独立 CPU 依赖见 [requirements.txt](requirements.txt)。

## 怎么运行

从仓库根目录执行：

```bash
python -m unittest discover -s part1-reproduction-20261025/code -p test_csa.py
python part1-reproduction-20261025/code/run.py --data /path/to/mnist.npz --output part1-reproduction-20261025/outputs/my-csa-run
```

下载原数据后替换路径，脚本校验 SHA；28×28 通过非重叠2×2均值下采样为14×14。训练只用训练均值中心化，分类为同协议1-NN。plan.json 固定训练子集 seeds、样本规模、rank 和停止规则。

## 结果与扩展

21 组 CSA/PCA 配对未显示 CSA 普遍胜过 PCA，但 CSA basis 更紧凑。相同 core 系数数目不等于相同总存储；同 latent 的最优训练重建应满足 PCA MSE≤CSA。

新增 rank/初始化/数据后另建 plan 和输出。保留正交、单调能量、能量恒等式与 PCA 下界检查；若推广高阶张量，需要真正实现并验证 mode 展开/乘积，现有代码只是二维。原 plan 的 USPS 字符串笔误与正确解释已在 RESULTS.md 记录。
