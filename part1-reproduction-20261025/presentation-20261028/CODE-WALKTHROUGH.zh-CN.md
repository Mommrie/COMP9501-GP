# 从论文公式到代码

入口：[材料包](README.md) · [演示与问答](DEMO-AND-QA.zh-CN.md)

## 1. 这份实现解决什么问题

训练图像组成 `x.shape = (N,H,W)`，例如 `(300,14,14)`。每张图仍是二维矩阵，`N` 只是样本数量。学习两个列正交投影 `U: (H,r1)` 和 `V: (W,r2)`，把图像映射为 `(r1,r2)` 的 core。

代码首先用**训练图像均值**中心化。以下用 `Z_i = X_i - mean`；不能让 test 参与均值或子空间拟合。

$$Y_i=U^T Z_i V,\qquad \hat X_i=U Y_i V^T+\overline X.$$

论文 Eq. (4) 是张量推广；Eq. (5) 最小化重建误差。我们的训练中心化是明确的实现约定。论文在 PCA 关系推导中也使用中心化假设（PDF p6，Claim 2）。

## 2. 为什么最大化 core 能量

列正交约束使投影重建是正交投影，因而：

$$\sum_i\|Z_i- U U^T Z_i V V^T\|_F^2
=\sum_i\|Z_i\|_F^2-\sum_i\|U^T Z_iV\|_F^2.$$

左边是 SSE，除以 `N*H*W` 才是代码报告的 pixel MSE。右边第一项固定，所以最小化 SSE 等价于最大化保留能量。训练曲线记录能量，不是 accuracy。

`TensorCSA.fit` 中 `energy()` 计算所有 core 元素的平方和；`diagnostics['energy_identity_error']` 独立比较“总能量 − core 能量”和真实重建残差。这个检查能发现矩阵方向或重建公式写错。

## 3. 固定 V，为什么更新 U 是特征分解

定义 `A_i = Z_i V`，其形状是 `H × r2`。固定 V 后：

$$\sum_i\|U^T A_i\|_F^2
=\operatorname{tr}\left(U^T B_U U\right),\quad
B_U=\sum_i A_i A_i^T.$$

约束 `U.T @ U = I` 时，Rayleigh–Ritz 原理给出：U 取 `B_U` 的前 r1 个特征向量。然后用**刚更新的 U**计算：

$$B_V=\sum_i (U^T Z_i)^T(U^T Z_i),$$

V 取其前 r2 个特征向量。这正是 Theorem 1 的二维特例；每个子问题条件最优不代表联合问题全局最优。

旧二维代码 [csa.py](../code/csa.py) 的对应关系：

```python
zv = z @ self.v                              # (N,H,r2)
self.u = leading(np.einsum('nhb,nkb->hk', zv, zv), self.rank)
uz = np.einsum('ha,nhw->naw', self.u, z)       # (N,r1,W)
self.v = leading(np.einsum('naw,nak->wk', uz, uz), self.rank)
```

这里 `einsum` 明确写出了对样本和其他维度求和的指标。`leading` 使用对称矩阵特征分解 `np.linalg.eigh`，按特征值从大到小选择列向量；用 SVD 的 PCA baseline 也求全局最优线性重建投影，不必显式建立巨大协方差。

## 4. 高阶版怎样工作

新 [csa_tensor.py](../code/csa_tensor.py) 使用 `TensorCSA(ranks=(r1,r2,...))`：

| 函数/位置 | 论文含义 | 需要留意 |
|---|---|---|
| `mode_product` | 单个 mode 的矩阵乘积 | mode 从 0 开始；真实数组轴为 mode+1，batch 轴不乘。 |
| `unfold_batch` | mode-k unfolding 与跨样本拼接 | 输出 `(m_k, N*其他维度)`；列有相关性，不是新增独立观测。 |
| `fit` 的 partial 循环 | Eq. (6)，压缩除当前 mode 以外的方向 | 循环内更新后的 factor 立即供后面的 mode 使用。 |
| `flat @ flat.T` | Theorem 1 的条件协方差 | 维数是 `m_k × m_k`。除以样本数不改变主特征向量。 |
| `_leading` | 条件最大能量子空间 | 按最大特征值取前 ranks[k] 列。 |
| `_project` / `transform` | Eq. (4) 的低维 core | 各 mode 乘 factor.T。 |
| `inverse_transform` | 把 core 还原到输入空间 | 各 mode 乘 factor，再加训练均值。 |
| `history` / `diagnostics` | Eq. (7) 的数值检查 | 数值容差内的单调性、正交性、能量恒等式。 |

例如 `(N,12,10,6) → (N,3,2,2)`，表示每张样本有三个空间/特征 mode。N 不是第四个要压缩的 mode。实现接口支持高阶输入，本轮测试实际覆盖 1/2/3 阶。

## 5. 一次完整拟合

1. 检查 rank、数据形状、有限数值；只接受实数，统一 float64。
2. 求训练均值并中心化。
3. 默认用每个 mode 未压缩协方差的主特征向量初始化；也支持给定随机 seed 的正交随机初始化。
4. 逐 mode 压缩其他方向、展开、计算条件协方差、更新该 mode 的 factor。
5. 每次 mode 更新后记录 captured energy；完整扫过全部 mode 称为一次 sweep。
6. 完整 sweep 的能量增益 `<= tol * max(abs(previous),1e-12)` 时停止，最多 100 sweeps。
7. 验证正交性、能量单调和重建恒等式；`converged=False` 表示触及上限，不应当作满足停止准则。

**停止准则与原文不同。** 原文 Procedure 1 在 PDF p4 印的是相邻投影相似度 `Tr[abs(U_new.T @ U_old)] / rank > 1-epsilon`，epsilon=0.001。本次视觉核对已确认是 `>`，旧材料所称“方向矛盾”不能继续沿用。我们用目标变化以避免特征向量任意符号/基旋转影响停止；不把这个改动说成原算法原样实现。两者也不能据迭代计数直接比较快慢。

## 6. 为什么说 CSA 是受约束的 PCA

对于 NumPy 默认 **C-order** 的行向量展开：

```python
(x-mean).reshape(N, -1) @ np.kron(U, V)
```

等于 `transform(x).reshape(N,-1)`。列优先数学 vec 常写 `V ⊗ U`，顺序取决于约定，不能混用。

CSA 只允许 Kronecker 可分离投影；PCA 可在所有 d 维线性子空间中选择。所以同一训练集、相同 d=r1*r2，最优 PCA 的**训练**重建误差不大于 CSA。测试误差和分类准确率没有这一必然排序。

三维测试把 `kron(kron(U0,U1),U2)` 与实际 mode 投影比较；一维 TensorCSA 与向量 PCA 的重建相同，验证论文 Claim 2。

## 7. 分类和指标怎样得到

`accuracy_1nn` 把每个 core 拉平，在 core 空间求欧氏距离最近的训练样本，返回其标签。CSA/PCA 拟合不读标签，标签用于分层取样和 1-NN。所有方法共用训练样本与 test IDs。

四个概念须分开：训练 SSE、平均 pixel MSE、最近邻 accuracy、存储标量数量。优化前者不等于优化分类。每图 RMSE `sqrt(sum_pixel_error)` 与 `sqrt(pixel_MSE)` 也差一个 `sqrt(H*W)`；原论文 Fig.5 的定义不能直接照搬本包纵轴数字。

## 8. 文件选择

- 要现场讲二维矩阵推导：用旧 `csa.py`，短且直接，原实验源码保持不变。
- 要展示论文任意阶机制：用新 `csa_tensor.py`，配三维合成验证，明确不是原文 Gabor benchmark。
- 要复现新图：用 `prepare_presentation.py`；拟合、指标、作图、CSV、数据/源码 SHA 都在一个入口。
- 要看实现是否算对：运行两个测试文件共 9 项检查；测试比较子空间重建而非要求特征向量元素完全相同。
