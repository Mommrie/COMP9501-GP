# Part 2 Proposal 与项目进度

**Activation-Aware Merging of LoRA Adapters for Multi-Task Learning**

COMP9501 · Group 13 · 进度更新：2026-11-11

[中文首页](README.zh-CN.md) · [搭建与复现](BUILD.zh-CN.md) · [扩展开发](docs/EXTENDING.zh-CN.md)

本文整理最初 proposal，随后对照现有实现与结果，末尾提供 timeline 和可继续更新的 TODO。最初 proposal 的预期不等于实验结论；代码可运行也不代表研究目标已经达到。

11 月 9 日完成 [新协议三 seed 结果与展示材料](PART2-RESULTS-V3-2026-11-09.zh-CN.md)：12 个新 teacher、同容量 CE/KL、固定权重基线、审计、图表和恢复验证。主指标 CE 0.7132、KL 0.6987、TIES 0.6146、individual 0.7695；学习系数改善当前固定基线，但 KL 未建立相对 CE 的平均优势。另见 [部署成本与标题诊断](PART2-FOLLOWUP-2026-11-09.zh-CN.md)、[独立五分区](INDEPENDENT-PROTOCOL-V3.zh-CN.md)、[相关工作](PART2-RELATED-WORK.zh-CN.md) 与 [实现 Q&A](PART2-IMPLEMENTATION-QA.zh-CN.md)。新最终区是原 train 的内部隔离分区，不是新外部盲测，不能与旧官方 validation 成绩直接作单因素归因。

11月11日进一步完成 [完整课程报告与展示素材](report-20261109/README.md)：L4五个FP32目标、真实/shuffled配对控制、任务性能导向router、oracle和帮助/损害分析，全部三seed及保存预测独立审计通过；英文PDF现扩展为15页，代码/结果精讲和逐页素材齐备。原预期并非全部成立：hidden未增益于CE，expert router未超过task-ID。恢复重跑不增加独立seed，Part1 CSA联系仅为概念动机。

来源为本地归档的 `COMP9501-Part2-Project-Proposal-Group13.docx`（2026-10-21 归档）。正文原文保留在下方，原 DOCX 与个人文件信息不随仓库分发。Proposal 是否已经正式提交、课程最终提交要求均（待核对）。

## 课程任务依据（与 proposal 的研究计划分开）

2026-10-28 对照课程原件 `Lec3 Performance Evaluation.pdf` 全文，并核看项目要求所在的第 2—4 页。课件留在课程资料目录，本仓库仅整理相关要求。

| 来源 | 课件明确说明 | 对本项目的含义 |
|---|---|---|
| 第 2 页：Group presentation，50% | 讲解指定的机器学习论文（非深度学习）；实现不强制，做实现有 bonus；15 minutes presentation & Q&A。 | Part 1 对应 CSA 论文讲解。补充复现可支持展示，但不能代替论文理解与问答。 |
| 第 2 页：Group project，50% | 实现自己的 machine/deep learning 研究想法；15 minutes presentation & Q&A；与 presentation 使用相同组员。 | Part 2 对应这里的 LoRA 合并研究，需要真实实现、实验与研究解释。课件没有指定必须采用哪一种模型或合并方法。 |
| 第 3 页：Group selection | Moodle 选组，9-02 开始，9-16 23:55 截止；未选则分配。换组须批准且双方全体组员同意。 | 这是组队行政安排，不是 Part 2 最终交付日期。 |
| 第 4 页：Paper selection | 阅读并实现/测试论文；可使用 CS GPU Farm；尽可能采用自己的测试/训练数据；通过 GitHub 向他人提供代码。 | 该页位于论文选题说明中；实现是否强制按第 2 页明确的“非必需、有 bonus”理解。课件没有写必须购买算力或自行采集数据。 |
| 第 4 页：Topic selection | 9-17 开始，9-23 23:59 截止，提交 top 7 choices，先到先得。 | 对应指定论文选题，不应当作自选研究项目的最终提交 deadline。CSA 在第 5 页论文列表中编号为 [4]。 |

**展示时间**：原文将 presentation 与 Q&A 写在同一个 15 minutes 描述中，没有另给问答时长。先按合计 15 分钟准备；不要自行扩成“讲 15 分钟，再额外问答”，具体分配仍可向课程核对。

**代码可访问性**：第 4 页要求通过 GitHub 让他人访问，没有明确要求仓库设为 public。当前 private repo 可供获授权的组员协作；老师/TA 的最终访问与提交方式仍需核对，仓库的公开性不在本次修改范围内。

**这份 PDF 未规定的内容**：Group 13 具体展示场次、proposal 截止/提交状态、正式报告是否必交及页数、幻灯片模板、最低 seed 数、必须完成哪些消融，均未在这 17 页中给出。第 9—16 页是 F-measure、AP、ROC、EER/AUC 教学内容，不等于要求本项目全部报告这些指标。下文的模型、任务、基线、消融和研究补强来自 proposal 或后续实验设计，不应全部理解为课程硬性要求。

## 1. 最初 proposal 想做什么

| 部分 | 原始计划的中文整理 |
|---|---|
| 问题 | 多个任务分别训练 LoRA adapter 后，直接平均权重可能造成任务干扰。希望用 adapter 的功能相似度指导合并或按输入路由。 |
| 动机与假设 | 部署每个任务各自的 adapter 有存储与管理成本；提出“激活相似的 adapter 更适合合并，冲突任务需要动态 gating”的假设。原文对既有方法的概括属于提案阶段动机，最终相关工作仍需核实。 |
| 方法 | 在多个任务上训练独立 LoRA；提取输入对应的 hidden activations，计算 adapter 两两相似度；尝试全局相似度驱动的静态加权合并，以及逐输入选择或组合 adapter 的动态 router。 |
| 模型与数据 | 小型 transformer（如 Qwen2.5-0.5B / TinyLlama）加 GLUE 子集，**或** ResNet/ViT 加四个图像分类数据集。这是备选路线，并非要求全部实现。 |
| 对照 | Individual LoRAs、simple averaging、Task Arithmetic、TIES、DARE。 |
| 指标与消融 | Average accuracy、worst-task accuracy、forgetting、OOD generalization；改变 similarity metric、merge coefficients、adapted layers。 |
| 预期 | Activation-aware 合并优于权重空间基线，尤其改善最差任务；动态路由减少干扰；交付可复现流程与定量分析。这些仍是待检验预期。 |

## 2. 当前实际采用的路线

- 采用 **Qwen2.5-0.5B + SST-2 / MRPC / RTE / CoLA**；TinyLlama 和图像路线暂未开展，也未作为当前必做项。
- Protocol v2：显式 Text / Sentence A / Sentence B 字段，最长 128 tokens；q/v LoRA，rank 8、alpha 16。每任务最多 4,096 条训练数据、3 epochs，使用原始 train 内隔离的 calibration 数据。
- 已完成训练 seeds **42 / 43 / 44**，共 **12 个任务 adapter**。三次重复共用数据划分，反映训练随机性，不覆盖不同数据划分或模型的波动。
- 合并对象是 backbone 的 LoRA 更新；**每个方法都使用已知任务对应的原分类头**。当前不是未知任务身份的统一分类器。
- 静态探索包括局部响应匹配、全局/逐层 CE 与 teacher-KL 系数拟合；动态探索包括 prototype、linear router 与 random / format / task-ID 对照。

复现入口与完整参数见 [BUILD](BUILD.zh-CN.md)、[配置目录](configs/README.zh-CN.md)。这些是实验过程中落实的方案，不是原 proposal 中已经写死的细节。

## 3. 原始目标与现有证据对照

| 原始目标 | 已有实现或实验 | 当前能得出的结论 / 剩余问题 |
|---|---|---|
| 训练多个任务 LoRA | 12 个 adapter；学习、梯度、分类头保存加载检查 | 已有可比较的任务银行；不等于模型/数据规模充分。 |
| Individual / average / TA / TIES / DARE | 三 seed 基线与 calibration 参数选择完成 | [基线报告](REPLICATION-2026-10-25.md)已包含主指标、mean accuracy、worst-task accuracy。TIES 为当前每矩阵实现，DARE 为 DARE+TA。 |
| 激活相似度能预测干扰 | 每 seed 六任务对；raw / Δactivation / ΔW cosine 对照 | 相关性跨 seed 不稳定，未证明 Δactivation 的独特预测价值；输入格式/来源混杂尚未排除。 |
| 静态 activation-aware 合并 | 局部响应匹配、全局/逐层系数原型与同容量 CE 对照 | 局部匹配没有稳定优于 uniform；逐层拟合改善了部分基线，但 teacher/activation 信号的独特贡献未成立。见[原型报告](PROTOTYPES-2026-10-25.md)。 |
| 按输入动态路由 | Prototype / linear router 与参照已运行 | Router 学的是数据源 task ID；分类头仍已知任务。高来源识别率不能解释为找到了最优 expert，也未建立部署优势。 |
| 减少真实标签依赖 | 固定 KL 轨迹改用 teacher-KL 选择 checkpoint | [选择控制](SELECTION-CONTROL-2026-10-25.md)保留改善，但 teacher 训练/原 checkpoint 选择用过标签；不是完全无监督。 |
| OOD generalization | IMDb / PAWS 各 512 条，固定五方法 × 三 seed | [迁移检查](TRANSFER-CHECK-2026-10-25.md)完成；未见稳定 KL 优势，PAWS 连 individual 都低于多数类参照。不是完整 OOD benchmark。 |
| Forgetting 与最差任务改善 | 已有逐任务指标、基线 worst accuracy、pair merge 损失 | 仍需统一定义本项目中的 forgetting（例如相对 individual 的下降），并汇总所有主要原型的最差任务表现。当前没有顺序任务训练实验。 |
| 系数 / 相似度 / adapted layers 消融 | 已有 global/layer、CE/KL、gain、raw/delta/weight 等比较 | 覆盖了部分消融；系统的 similarity、格式控制与 q/v 之外 adapted layers 比较尚未完成。 |
| 可复现与组员使用 | 代码、配置、结果 JSON、图表、中英文 Colab 与扩展指南 | 本地打包/恢复检查已通过；checkpoint 云共享地址与队友新账号的实际 Colab 复现仍待完成。 |

### 读结果时需要保留的边界

现有四任务主指标是 SST-2 accuracy、MRPC (accuracy+F1)/2、RTE accuracy、CoLA MCC 的平均，**不是官方完整 GLUE 分数，也不是平均 accuracy**。下表 ± 为三训练 seed 的样本标准差，不是置信区间。

| 配置 | 四任务主指标 mean ± SD |
|---|---:|
| Individual adapters | 0.7684 ± 0.0027 |
| Average | 0.6063 ± 0.0219 |
| Uniform mixture，gain 2 / 对应 TA 配置 | 0.6365 ± 0.0365 |
| TIES（每矩阵） | 0.6526 ± 0.0284 |
| DARE + TA | 0.6323 ± 0.0295 |
| 逐层 CE 拟合，真实标签选择 | 0.6997 ± 0.0124 |
| 逐层 teacher-KL 拟合，teacher-KL 选择 | 0.6996 ± 0.0126 |

数值来自[三 seed 基线](REPLICATION-2026-10-25.md)、[原型](PROTOTYPES-2026-10-25.md)和[后续选择控制](SELECTION-CONTROL-2026-10-25.md)。CE 与 KL 的信息使用方式不同；不能仅凭上述均值宣称 KL 更好。逐层拟合仍低于 individual。源 validation 已在探索中反复查看；系数 fit/selection 半区隔离不等于 teacher 训练到最终选择的全流程独立 holdout。

Δactivation 相似度与 pair merge accuracy 损失的 Spearman ρ 在三个 seed 分别为 **−0.6571 / −0.4857 / +0.0857**。六对共享 adapter，不能把三 seed 的 18 个观测当作 18 个独立任务对。Linear router 的旧来源识别率为 98.31%；每批需一次 base 特征前向，再按不同 expert 分组执行，不能概括为永远两次。当前已知 task ID 直接选 adapter 只需一次。[11 月 9 日 A100 成本实测](PART2-FOLLOWUP-2026-11-09.zh-CN.md) 已补延迟、吞吐、显存与存储；GPU 缓存降低状态传输成本，但仍有额外前向与缓存开销，未建立当前协议下的部署优势。

因此，目前可以报告“实现、对照、诊断与失败条件”，还不能将原 proposal 中“优于权重空间方法、改善最差任务、减少干扰”的预期整体记为完成。相关工作定位与最终主张需要一起收敛。

## 4. 原始 proposal 正文（保留英文）

以下为原 DOCX 的项目正文；内容保持原意和原文，便于对照，不作为更新后的实验结论。原文件没有时间表或 TODO。

### Problem Statement

When multiple LoRA adapters are trained for different tasks, simply averaging their weights often causes task interference and degrades performance. It remains unclear how to merge or route adapters dynamically based on their functional similarity.

### Motivation

LoRA is efficient and widely used, but deploying one adapter per task is costly. Model merging offers a lightweight alternative, yet existing methods like Task Arithmetic and TIES use weight-space heuristics and ignore input-dependent conflicts. We hypothesize that adapters with similar activations can be safely merged, while conflicting tasks require dynamic gating.

### Methodology

We will train separate LoRA adapters on multiple tasks. For each validation input, we extract hidden activations and compute pairwise activation similarity between adapters. We then design an activation-aware merging strategy: a static weighted merge based on global similarity, and a dynamic router that selects or combines adapters per input.

### Experimental Setup

We plan to use small transformer models (e.g., Qwen2.5-0.5B or TinyLlama) on GLUE subsets, or ResNet/ViT on four image classification datasets. Baselines include individual LoRAs, simple averaging, Task Arithmetic, TIES, and DARE. Metrics: average accuracy, worst-task accuracy, forgetting, and OOD generalization. Ablations will vary similarity metrics, merge coefficients, and adapted layers.

### Expected Outcomes

We expect activation-aware merging to outperform weight-space baselines, especially on the worst task, and dynamic routing to reduce interference. We will also provide a reproducible pipeline and quantitative analysis.

## 5. Timeline（建议安排，待组内调整）

以下是 **2026-10-28 整理的工作计划**，不是原 proposal 的时间表，也不是新确认的课程 deadline。展示日期来自另一份课件 `Lec0 Introduction.pdf` 第 3 页：Part 1 为 **11-04 / 11-11**，Part 2 为 **11-18 / 11-25**；本次提供的 Lec3 没有展示日程。Group 13 具体场次、年份与星期映射、提交形式均（待核对）。先按最早场次留出准备时间，不把额外一周当作已获延期。

| 日期 / 阶段 | 计划或已发生事项 | 可检查的产物 / 状态 |
|---|---|---|
| 10-21（已发生） | 原始 Part 2 proposal 本地归档 | 一页提案；正式提交状态（待核对）。 |
| 10-25（已发生） | Protocol v2、三 seed 基线、诊断、静态/路由原型、选择控制与迁移检查 | 本仓库对应报告、配置和结果；同日也完成 Part 1 CSA 补充实验与展示提纲。 |
| 10-28（本次） | 整理共享工程、中英文使用指南与 proposal 进度对照 | 本文、中文入口、Colab 启动与扩展说明；团队功能仍在 feat/team-colab。 |
| 10-28—10-30（建议） | 核对具体场次/提交要求；组内确定最终问题、必要消融与分工 | 一份收敛后的方法与对照清单；启动 Part 1 正式幻灯片。 |
| 10-31—11-02（建议） | 优先做能区分 activation 信号与容量/格式因素的关键控制；整理指标 | 固定协议、必要实验、逐任务/最差任务/相对 individual 损失汇总；Part 1 首次计时演练。 |
| 11-03—11-04（建议） | 为最早 Part 1 场次冻结材料、准备问答 | CSA 正式幻灯片与讲稿；若实际为 11-11，再利用缓冲修订。 |
| 11-05—11-10（建议） | 完成已选关键消融、成本测量与复现；准备较晚 Part 1 场次 | 结果表、局限、checkpoint 共享与一次队友复现；停止无界方法搜索。 |
| 11-11—11-14（建议） | 完成 Part 2 叙事、图表、引用和正式展示材料 | 可评阅的幻灯片；若课程另要求报告，按核实要求准备。 |
| 11-15—11-17（建议） | 按最早 Part 2 场次冻结结论与复现包、计时演练 | 可复现版本、最终图表、按展示与 Q&A 合计 15 分钟预备。 |
| 11-18 或 11-25（待核对） | 按 Group 13 的实际 Part 2 场次交付 | 正式展示/提交；若为晚场，优先补稳健性与排错。 |

Part 1 与本文 Part 2 是不同交付；CSA 实验完成不等于 Part 1 已展示。其[结果](part1-reproduction-20261025/RESULTS.md)与[展示提纲](part1-reproduction-20261025/PRESENTATION-OUTLINE.md)可直接复用，但正式幻灯片、实际分工和计时演练尚未完成。

## 6. TODO / 完成记录

勾选表示对应产物或检查已完成，不表示原始研究假设成立。下面同时包含课程交付、原 proposal 计划和研究质量补强；除上文列明的课件要求外，优先级是项目建议，可由组内按最终主张与时间调整。可以在条目后补结果链接或简短日期，不必另建复杂格式。

### 已完成

- [x] 整理原始 proposal，并保留英文正文与中文说明。
- [x] 核对 Lec3 的课程任务要求，区分两部分考核、选题日期、代码访问与后续研究建议。
- [x] 固定当前 Qwen2.5-0.5B、四任务、输入字段、任务头和训练/评价协议。
- [x] 完成学习/梯度/头部保存加载与合并 logits 等工程检查。
- [x] 完成 42/43/44 三训练 seed 的 12 个任务 adapter。
- [x] 完成 individual / average / TA / TIES / DARE+TA 基线，以及基线 mean / worst-task accuracy。
- [x] 完成六任务对干扰与 raw / Δactivation / ΔW 相似度诊断，保留不稳定结果。
- [x] 实现并运行局部响应匹配、global/layer CE/KL 原型与同容量 CE 对照。
- [x] 实现并运行 prototype / linear router，加入 random / format / task-ID 参照。
- [x] 完成固定 KL 轨迹的 teacher-KL selection 控制。
- [x] 完成固定 IMDb / PAWS 子集迁移检查，保留负结果与截断限制。
- [x] 提供配置、结果 JSON、图表、中文说明与队友扩展入口。
- [x] 完成 Colab 源码打包、checkpoint 恢复与结果导出的本地验证。

- [x] 2026-11-11 完成直接hidden、shuffled、global/layer及best-expert/source路由对照，完整三seed预测、原始选择、图表、独立审计和恢复重复比较齐备。
- [x] 2026-11-11 完成英文LaTeX/PDF报告、中文机制/数值精讲、逐页展示素材和机制Q&A；双语notebook增加v3入口并修复跨平台ZIP快照，28项代码测试通过。

### 优先补齐（交付与研究建议）

- [ ] 核对 Group 13 的 Part 1/2 场次、proposal 提交状态及正式提交物要求。
- [ ] 组内确定最终研究主线与分工：保留静态/动态哪些部分，如何呈现未支持的假设。
- [x] 2026-11-09 核对 AdaMerging、RegMean、KnOTS 的直接相关方法与引用，说明差异与未复现范围；见 [相关工作](PART2-RELATED-WORK.zh-CN.md)。
- [x] 2026-11-09 固定本轮各方法的信息权限与 task-aware head；未知任务路由不纳入已完成主张，需要另定义输出/头协议。
- [x] 2026-11-09 完成五分区、内容 tuple 分组与新 teacher 三 seed 重训，隔离训练、teacher 选择、系数拟合/选择和最终评价；不把旧 validation 或本轮内部最终区称新外部盲测。
- [x] 2026-11-09 完成本轮已选控制：同容量同步数 CE/KL、冻结 router 标题扰动；隐藏层 activation 的独特贡献和完全去来源风格未建立，未覆盖范围写入报告。
- [x] 2026-11-09 汇总 v3 八种方法逐任务、mean/worst accuracy、native 指标与相对 individual drop；三 seed CSV/JSON 与图表见 [结果报告](PART2-RESULTS-V3-2026-11-09.zh-CN.md)，不称顺序学习 forgetting。
- [x] 2026-11-09 测量部署成本：A100 固定输入下的延迟、吞吐、显存与 adapter/合并模型存储，计入 router 的额外前向；见 [测量范围与结果](PART2-FOLLOWUP-2026-11-09.zh-CN.md)。
- [ ] 设置 checkpoint 的授权共享位置，并由队友自己的账号完成一次 Colab 恢复和复评。
- [ ] 完成 Part 2 正式幻灯片、结论/局限、计时演练与问答；按核实要求准备其他交付。
- [ ] 完成 Part 1 正式幻灯片、实际分工与计时演练（独立于 Part 2）。
- [ ] 完成实际展示/提交并记录回执；目前不能勾选课程项目已交付。

### 按结论与时间选做

- [ ] 在固定协议下补更多 similarity / adapted layers 消融；当前只完成其中一部分。
- [ ] 若最终结论确需更广泛外推，再安排新模型、新数据划分或新的独立 OOD 目标。
- [ ] 需要时继续完善动态路由；先明确它在当前 task-aware 参照之外解决的问题。
- [ ] TinyLlama / ResNet / ViT / 图像路线仅在组内决定扩展时开展，不作为原 proposal 的全部必做项。

- [x] 2026-11-11 第二轮补强完成：λ=0/.01/.1/1 matched aligned/shuffled、confidence/benefit gates、锁定后2,580条官方validation复评；完整预测审计与28项云端测试通过。CE 0.7050，selected aligned 0.7106，shuffled 0.7107；两类gate三seed均选always-own。未建立独特activation优势，保留历史探索边界。详见 [结果与解释](report-20261109/ROBUSTNESS-RESULTS.zh-CN.md)。
