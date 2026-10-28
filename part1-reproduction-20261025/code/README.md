# CSA numerical demonstration

- `csa.py`：二维 CSA / PCA；训练均值中心化；Gauss–Seidel 式交替更新；显式保存每个坐标的 captured energy。
- `test_csa.py`：正交性、单调性、能量恒等式、PCA 可行域下界与可分离数据精确重建检查。
- `run.py`：校验原数据 SHA，按冻结方案比较训练重建 / 测试重建 / 1-NN，并保存实际样本 ID 与模型。
- `summarize.py`：汇总 JSON、Markdown 表与两张结果图；不选择测试表现最好的配置。

在本目录的上一级运行：

```powershell
python -m unittest discover -s code -p test_csa.py
python code/run.py --data path/to/mnist.npz --output outputs/mnist-v1
```

依赖 NumPy；数据和模型输出不纳入可分享代码。向量化使用 NumPy C-order，Kronecker 投影为 `kron(U,V)`；不把论文的符号向量化顺序强套到另一种数组约定。
