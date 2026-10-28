# Part 1 CSA：二维原理复现与展示准备

本目录为独立的补充实验，不改原论文或旧导读卡。实现论文二维特例：中心化图像 `X`，列正交 `U,V`，core=`U.T @ X @ V`，重建=`U @ core @ V.T`。交替求条件协方差的主特征向量，对应原文 Theorem 1 / Procedure 1 的坐标更新。

- [冻结实验方案](plan.json)：运行前固定维数、数据与指标；不是 2008 年原表的数值复现。
- [代码](code/README.md)：NumPy 实现、数值检查、MNIST 受控比较。
- [结果与限制](RESULTS.md)、[原始凭据](artifacts/README.md)：21 组比较完成；没有显示 CSA 在该设置下普遍胜过 PCA。
- [12 分钟讲解稿与问答](PRESENTATION-OUTLINE.md)：待组员审阅、分工与计时，不是已提交 slides。
- `outputs/`：忽略提交的数值模型与运行结果；轻量结果另行归档。

数据为 Keras 官方 MNIST 镜像，28×28 以非重叠 2×2 均值下采样成 14×14。仅复用公开原数据缓存；不使用 DATA8014 的 nuisance channel、划分或模型结果。代码不分发课程 PDF 或受限论文。
