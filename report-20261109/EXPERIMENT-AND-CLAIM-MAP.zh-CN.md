# 从 proposal 到课程报告：实验与结论对应

2026-11-09。目标是完成一份能独立阅读、能支撑15分钟展示及问答的课程研究报告，不以得到预期正结果作为“完成”的条件。课程是否要求正式书面报告、页数及Group13场次仍待核对；报告本身按完整科研结构准备。

运行资源更新：用户提示A100未分配，重试实测仍为NVIDIA L4（约23GB）。新增FP32 controls使用这张云端L4，五种目标使用相同设备；旧v3训练及成本实测的A100记录保持原样。复用校验备份恢复12个teacher，不把恢复说成重新训练。模型脚本要求CUDA且至少20GiB显存，CPU本机不运行。

## 1. 对照最初 proposal

| 原始承诺 | 已有证据 | 本轮加强 |
|---|---|---|
| 多任务LoRA平均会干扰 | v3三seed individual/average及逐任务损失 | 正文明确问题与真实共享base参照，避免夸大存储动机 |
| 相似activation对应安全合并 | 旧v2六任务对×三seed，相关方向不稳定 | 将它作为独立RQ报告，区分raw/Δactivation/ΔW及重叠pair的统计限制 |
| activation-aware静态合并 | 局部响应原型、logit KL、同容量CE；v3隔离协议 | 直接匹配最终隐藏层响应；加入同精度CE和打乱响应对照 |
| dynamic routing选择有用adapter | 旧source router及标题/成本诊断 | 拟合逐样本最小GT-NLL expert，比较task-ID、来源router、oracle准确率上界 |
| Average/TA/TIES/DARE及individual | v3固定17候选网格、三seed完整结果 | 主表保留原结果，扩展另表，不事后更改网格 |
| 平均、最差任务、forgetting、OOD | mean/worst accuracy、相对individual损失；旧IMDb/PAWS负结果 | 明确性能损失不等于顺序学习forgetting；不把旧OOD混入新teacher结果 |
| similarity/coefficients/layers消融 | 旧raw/Δ/weight、global/layer探索 | 全局5标量vs逐层97标量、真实vs打乱输入响应；未做q-only/v-only等模块重训 |
| 可复现、解释与展示 | v3源代码/ID/配置/权重备份、Q&A | 完整英文报告与PDF、逐页素材、讲稿和结论边界 |

## 2. 三个研究问题

**RQ1：功能相似度能否可靠预测合并损失？** 使用旧v2原始诊断，报告三seed全部相关值，六pair共享任务，不能作独立样本显著性推断。它与新v3最终区不是同一数据协议。

**RQ2：在控制容量、精度和监督后，隐藏响应是否带来额外帮助？** v3证明学习系数优于当前固定网格，但logit KL未平均胜过CE。本轮直接研究hidden signal，避免用最终输出KL代替中间表示假设。

**RQ3：识别来源与选择有用expert是否等价？** 已知task head的协议中，task ID直接选择own adapter本来就可用。新路由需与此比较，并报告GT oracle准确率上限及离线expert评估代价。

## 3. 新增静态对照

复用新v3 teacher，不重新训练LoRA/head。保持128条/任务fit、128条selection、最终256条，使用既有manifest SHA。五个新profile均FP32，24层版本97个标量，全局版本5个；120steps、batch8、lr0.03、相同初始化/任务顺序/抽样。CE与teacher hidden的组合权重预先固定为1，不用final搜索它。

令 `h_t(x)` 是teacher最后隐藏层的mask-aware token mean，`h_0(x)` 为base，同一输入响应 `d_t=h_t-h_0`。学生响应 `d_s=h_merge-h_0`。隐藏损失为逐例 `mean((d_s-d_t)^2) / max(mean(d_t^2),1e-4)` 再取batch均值。两边减同一base使分子等价于raw hidden MSE；减base的作用在目标能量归一化，不能伪称是全新的差分距离。

| Profile | 拟合目标 | 参数 | 要隔离的因素 |
|---|---|---:|---|
| layer_ce_fp32 | CE +相同系数正则 | 97 | 新扩展的同精度参照；与旧v3 BF16 CE另列 |
| layer_hidden | normalized hidden MSE +正则 | 97 | 只用teacher响应的系数更新；selection仍用标签 |
| layer_ce_hidden | CE +真实配对hidden MSE +正则 | 97 | 与同精度CE对照，测试增加响应目标的效果 |
| layer_ce_shuffled | CE +任务内打乱hidden delta +正则 | 97 | 固定无不动点permutation，保留目标分布、破坏逐输入对应 |
| global_hidden | 与layer_hidden相同 | 5 | 全局与逐层系数容量/位置自由度 |

新扩展的分区内访问仍隔离，但**v3 final分数已经查看**，新增实验是探索性扩展，不能把同一final重新称盲测。预固定配置、按selection选step只能阻止本轮直接final调参，不能抹去提出实验时已有的知识。

FP32用于降低小Δ响应的数值混淆；原v3 BF16训练的CE/KL保留原样，所以新旧行不构成仅改变损失的一因素对照。真实响应与打乱响应使用完全相同的新精度和优化预算；打乱只能检验输入配对作用，不能排除所有数据风格混杂。

## 4. 新增路由对照

每个任务的分类头始终由已知task ID提供，四个expert只改变backbone ΔW。先在fit/selection上缓存每个expert的分类logits及base mean features。根据fit GT求每例最小NLL expert作为拟合目标；这显式使用fit类标签及全部expert离线前向，不能称无标签或零成本路由。

- **task_id**：选本任务adapter，单次backbone的可用参照。
- **source_router**：base feature预测来源task ID，保留旧原型的目标；本轮按selection任务主指标选C。
- **expert_router**：标准化base feature加已知task one-hot，拟合最小NLL expert；相同C候选0.1/1/10，用selection任务主指标选C。
- **random**：固定seed随机expert参照。
- **oracle**：final GT逐样本选择真类概率最高的expert，给出当前bank/head的accuracy ceiling；**不是可部署方法**。它不一定是所有native metric的最优路由，因此只将accuracy称上界。

真实路由和静态混合一样，先锁定selection文件再读取本轮final。缓存四expert的离线计算只用于分析，不是部署时只跑所选expert的实测；部署开销另引用已有A100实测，不能将oracle或离线缓存当成免费推理。

## 5. 报告和presentation应支持的结论

允许由结果决定：真实hidden目标是否优于同精度CE、是否胜过shuffled；逐层是否优于全局；任务性能导向路由是否接近task-ID参照、与oracle还差多少。即使全部没有提升，也可完整报告“原假设在受控小模型场景下的失败条件”，不能为契合proposal预期改写结果。

报告固定保留：摘要、原假设、相关工作、数学方法、数据/信息权限、主结果、直接activation消融、路由及oracle、成本与OOD、讨论、局限、复现附录、结论和引用。presentation按约11分钟内容加4分钟QA准备，这是工作分配建议，课程仅明确合计15分钟。
