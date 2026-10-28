# 输入路由原型

[原有说明](README.md) · [中文首页](../../README.zh-CN.md) · [扩展开发](../../docs/EXTENDING.zh-CN.md)

Router 使用 frozen base 的 pooled feature 预测数据源 task ID。比较 random、format-only、prototype、linear 与直接 task-ID 选择；原分类头仍已知，因此高 source accuracy 不等于学到了逐样本最优 expert。

```bash
python code/router_v1/run.py --config configs/protocol-v2-a100.json --plan configs/router-v1.json --output outputs/my-router-v1
```

三个参数都必需。Runner 固定读取 seeds42/43/44 的标准 bank 路径；只有一个 bank 时不能直接跑完整脚本。输出包括 route/选择与逐 seed validation，目标特征可能缓存为较大文件，留在 outputs 而不进 Git。

## 扩展注意

StandardScaler 只能 fit 在拟合半区；C 等参数只用选择半区。更换任务数、类别、seed 或 bank 布局要修改原型个数、logits 维数与路径接口。

Feature router 需要一次额外 base 前向，另执行所选 adapter。新增机制应报告真实成本，并明确是否已知任务头。当前协议下直接 task-ID 已可选原 adapter，不要把它藏掉后声称部署优势。
