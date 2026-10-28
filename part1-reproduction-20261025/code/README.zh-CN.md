# CSA 数值代码

[原有说明](README.md) · [中文首页](../../README.zh-CN.md) · [扩展开发](../../docs/EXTENDING.zh-CN.md)

csa.py 包含二维 CSA 与向量 PCA，run.py 读取数据/方案并保存样本 ID、模型与数值结果，summarize.py 从归档 JSON 汇总图表，test_csa.py 做数学检查。依赖 NumPy，汇总绘图另需 Matplotlib。

## 核心计算

中心化 X 的 core=UᵀXV，重建=U core Vᵀ。固定 V 更新行协方差主特征向量，再用新 U 更新 V；记录每次坐标更新的 captured energy。代码使用 C-order，Kronecker 顺序与数组约定配套。

从根目录运行模块检查：

```bash
python -m unittest discover -s part1-reproduction-20261025/code -p test_csa.py
```

## 扩展与验收

改迭代或初始化时保留列正交、能量不减、重建能量恒等式；精确可分离数据应重建到数值误差。匹配 latent 时 PCA 训练 MSE 不应高于 CSA。目标收敛不保证全局最优或参数唯一。

当前 run.py 的14×14下采样与平衡取样均有具体实现，改 plan.side 并不会自动支持任意图像大小；需要连同 downsampling、basis 计数、维数上限与绘图标签一起修改。
