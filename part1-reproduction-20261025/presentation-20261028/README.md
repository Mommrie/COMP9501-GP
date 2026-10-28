# Part 1 code and evidence pack / 代码与展示材料包

面向负责 CSA 实现、代码讲解和制作 PPT 的组员。2026-10-28 整理，**不是幻灯片，也不表示已完成展示**。课程要求为论文讲解，实现为 bonus；本包提供可复现的补充证据。

## 从哪里开始

| 你要做的事 | 材料 |
|---|---|
| 先理解代码做了什么 | [代码讲解与公式对应](CODE-WALKTHROUGH.zh-CN.md) |
| 选择能放入 PPT 的图 | [图表目录、图注与讲述建议](FIGURE-GUIDE.zh-CN.md) |
| 对照原论文，避免混用实验结论 | [原文证据与实现边界](PAPER-EVIDENCE.zh-CN.md) |
| 阅读新实验与完整数字 | [实验解释](EXPERIMENTS.zh-CN.md)、[完整结果表](figures/results-table.md)、[逐 seed CSV](figures/results.csv) |
| 准备代码演示和问答 | [演示路线与 Q&A](DEMO-AND-QA.zh-CN.md) |
| 实际运行 | 本页下方命令；[实验方案](plan.json) |

图均提供 **PNG 和可编辑文字的 SVG**，英文图面便于直接用于英文 slides，中文解释见配套文档。[图目录](figures/README.md)中列全 10 张图；原论文表格重绘和我们的新实验在文件名、标题和图注中均区分。

## 代码和新结果

- 新增 [csa_tensor.py](../code/csa_tensor.py)：支持每张样本为一维、二维或更高阶实数张量，保留 batch 轴；逐 mode 条件特征分解。任意阶接口不代表已测试所有阶数，本轮验证覆盖 1D/2D/3D。
- 旧 [csa.py](../code/csa.py)、旧 runner 和 2026-10-25 的原始科学结果保留；新二维结果在六个重合配置与旧结果一致到数值误差。
- 21 个新配置（三训练子集 × 七个 rank），每组报告 CSA、相同 latent 的 PCA，以及存储预算内的 PCA；另有九种初始化检查与独立的三维合成数据验证。
- 数学与接口测试共 9 项通过；高阶扩展由显式指标求和、Kronecker 恒等式、PCA 特例、重建能量与 held-out 精确重建交叉验证。
- MNIST 新扩展使用已被探索过的测试集，不能作为新盲测。没有完成原文的 Gabor / LDA / 人脸实验复现。

## 运行（从仓库根目录）

Part 1 独立于根目录的 LoRA/Colab 环境。可使用已有兼容环境，或创建自己的虚拟环境后安装：

```bash
python -m pip install -r part1-reproduction-20261025/requirements.txt
python -m unittest discover -s part1-reproduction-20261025/code -p "test_csa*.py" -v
python part1-reproduction-20261025/code/prepare_presentation.py --download --output part1-reproduction-20261025/outputs/presentation-my-run
```

`--download` 仅在默认缓存不存在时从官方 Keras 数据镜像下载 MNIST；随后检查固定 SHA-256。已有数据可用 `--data /path/to/mnist.npz`。输出目录必须尚不存在，脚本不替换旧运行。所有拟合/绘图仅需 CPU。本轮在 Python 3.9、NumPy 1.26.4、Matplotlib 3.9.2 上验证；环境版本和源码哈希记录在结果 JSON。

可用 `--plan path/to/new-plan.json` 运行另一个方案；当前 runner 明确只实现 28×28→14×14，修改 side 不会自动实现新预处理。新方案应保留配置、运行结果和完整报告，不根据 test 表现只留优胜点。

输出含 10 组 PNG/SVG、JSON、CSV、Markdown 表和 SHA 清单；不保存训练图像全集或模型权重。Git 里只保留展示所需图表与轻量凭据，MNIST 缓存不上传。[校验记录](VALIDATION.md)说明实际检查。

## 给 PPT 制作者的建议

时间紧时先取：**01 流程 → 02 重建 → 06 存储 → 07 收敛**；讲实验结论再取 **04 相同 latent / 05 存储预算**。原论文实验可选 **10 Table IV 重绘**，务必保留来源与“作者报告的最佳配置”说明。08 初始化和 09 三维验证适合作为附录。

既有 [12 分钟论文提纲](../PRESENTATION-OUTLINE.md)仍可供总体结构参考；本包补充的是实现及证据，不要求负责代码的组员制作整份 PPT。
