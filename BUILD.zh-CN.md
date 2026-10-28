# 环境搭建与复现：中文版

[中文首页](README.zh-CN.md) · [English](BUILD.md) · [扩展开发](docs/EXTENDING.zh-CN.md)

以下命令均在仓库根目录执行。训练依赖要求 Python ≥3.11；归档 GPU 实验使用 Python 3.13.16、torch 2.11.0+cu130、A100 40 GB。保留 Colab 自带 torch/CUDA，不照着旧日志替换整个 CUDA 环境。实际 import 版本见 [环境记录](artifacts/effective-runtime-2026-10-25.json)。

## 1 选择运行方式

- 直接用 Colab：打开 [中文团队 notebook](notebooks/team_colab.zh-CN.ipynb)，首次运行会校验源码快照、安装依赖并做离线检查。训练、复评和导出均由开关显式启用；不需要 MCP 或 runtime GitHub token。
- 本地/其他服务器：先用自己的账号 clone private repo，进入目录后安装 requirements.txt；需能访问公开模型和数据发布方。GitHub/Colab/Drive 权限分别管理，见 [协作说明](COLLABORATION.md)。

```bash
python -m pip install -r requirements.txt
```

启动训练 Python 之前设置三个环境变量。PowerShell：

```powershell
$env:USE_TF='0'
$env:USE_FLAX='0'
$env:TOKENIZERS_PARALLELISM='false'
```

Linux / Colab shell：

```bash
export USE_TF=0
export USE_FLAX=0
export TOKENIZERS_PARALLELISM=false
```

## 2 先检查，再训练

```bash
python scripts/check_repository.py
python -m unittest discover -s code -p test_math.py
python -m unittest discover -s code -p test_protocol.py
```

仓库检查只用标准库，帮助查语法、JSON、链接和明显的凭据/路径问题；按需要运行即可，不作为协作格式门槛。数值检查需要 torch，各模块还有对应测试。

## 3 建立或恢复 adapter bank

只想复评已有实验时，先按 [权重共享指南](assets/README.zh-CN.md) 恢复 12 个 adapter，使用 notebook 的 RUN_EVALUATION；无需重新训练全部任务。

要从头训练，输出目录必须尚不存在：

```bash
python code/run_bank.py --config configs/protocol-v2-a100.json --seed 42 --output outputs/bank-v2-a100-s42
python code/run_bank.py --config configs/protocol-v2-a100.json --seed 43 --output outputs/replication-s43-s44/bank-s43
python code/run_bank.py --config configs/protocol-v2-a100.json --seed 44 --output outputs/replication-s43-s44/bank-s44
```

一个 bank 包含四任务训练、individual/average 评价和共同输入 probe，状态写在 status.json。默认四任务为 SST-2/MRPC/RTE/CoLA，头部已知且均为二分类。模型 commit 固定；数据分区 ID/fingerprint 被记录，但默认 GLUE 配置没有显式 dataset_revision，不要把它描述成完整数据版本锁定。

## 4 按依赖顺序跑后续研究

| 模块 | 需要先有什么 | 入口 |
|---|---|---|
| 合并基线与六任务对 | 单 seed bank | merge_study/run.py，带 config、plan、bank、output |
| 局部响应匹配 | 三 seed bank，或单 seed CLI | functional_merge/run.py / run_all.py |
| 静态 CE/KL 系数 | 三 seed bank | coefficient_merge/run_all.py |
| 同容量 layer CE | 三 seed bank | coefficient_merge/run_layer_ce_all.py |
| Feature router | 三 seed bank | router_v1/run.py，必须传三个 CLI 参数 |
| 选择标签控制 | coefficient-v1 的原 history | selection_control/run.py |
| IMDb/PAWS 固定迁移 | bank、CE 控制、selection-control | transfer_check/run.py |

Router 的完整示例：

```bash
python code/router_v1/run.py --config configs/protocol-v2-a100.json --plan configs/router-v1.json --output outputs/router-review-v1
```

其他完整命令见 [代码模块中文导航](code/README.zh-CN.md)。部分 runner 固定 outputs 子目录；恢复包可能已含这些目录。看到 FileExistsError 时保留旧结果，改用支持独立路径的单次 CLI，或先为 runner 增加路径参数；不要为了重跑删除旧证据。

## 5 Part 1 CSA

```bash
python -m unittest discover -s part1-reproduction-20261025/code -p test_csa.py
python part1-reproduction-20261025/code/run.py --data /path/to/mnist.npz --output part1-reproduction-20261025/outputs/my-mnist-v1
```

将数据路径改成实际下载位置，脚本会校验 SHA。它是二维灰度 MNIST 补充演示，不能替代原文人脸/Gabor-MNIST 表格；细节见 [CSA 中文指南](part1-reproduction-20261025/README.zh-CN.md)。

## 6 保存、排错与扩展

- outputs/ 是运行产生的内容；artifacts/ 是已经审阅的归档证据。新运行使用新名字，报告注明代码/快照、配置、数据版本、seed 与选择规则。
- ModuleNotFoundError：核对当前 Python 和固定依赖，避免把依赖装进另一个 kernel。Colab 先执行环境 cell，再启动子进程脚本。
- 显存不足：先降低 batch，并记录梯度累积/有效 batch 的变化；不直接拿不同预算结果作方法优势。GPU 分配随账号与可用资源变化。
- 缺 bank/history：先看上表依赖和 assets manifest；仅有 summary JSON 无法重新计算真实模型评价。
- 新任务/模型/方法：先读 [扩展开发指南](docs/EXTENDING.zh-CN.md) 的固定假设表；不要直接改 tasks 后启动完整训练。
