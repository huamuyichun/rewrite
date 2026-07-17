# Graph-level Latency Estimator for Rewrite Plan Selection

## 1. 课题定位

本课题关注深度学习推理图优化中的 `rewrite plan selection` 问题。给定一张原始计算图及若干语义等价的候选 rewrite plans，目标是在真实执行前预测不同候选计划的图级执行代价，并选择最终最优或接近最优的计划。与传统基于规则、局部 pattern 或单算子代价加和的方法不同，本课题强调：rewrite 的真实性能收益往往由图级结构相互作用决定，因此需要显式建模全图或局部作用域图的结构特征。

从研究性质上看，这不是一个“给编译器打补丁”的工程问题，而是一个“在等价计算图空间中进行结构化学习决策”的问题。其核心不在于单纯拟合 latency，而在于服务 plan selection：在有限搜索预算下，选到更好的 rewrite plan。

## 2. 研究背景与问题动机

现有推理系统与编译器在图优化阶段通常会进行 fusion、layout movement、transpose sinking、algebraic reassociation、subgraph substitution 等改写。对同一段计算图，往往存在多种合法且语义等价的 rewrite 路径；但这些路径在最终执行中的 latency 并不相同。

困难在于：

1. 某个局部 rewrite 的收益不只由该子图自身决定，还依赖前后邻域、layout 传播、后续 fusion 机会、memory movement 和 lowering 结果。
2. 单算子 latency 或 FLOPs/bytes 之和通常不足以反映 rewrite 对端到端执行代价的真实影响。
3. 候选 rewrite plans 数量随图规模迅速增长，完全依赖真实编译与 profile 的成本较高。

因此，本课题要解决的问题是：

> 如何利用图表示学习，预测语义等价候选 rewrite plans 的图级执行效果，并在有限搜索预算下选出更优计划？

## 3. 课题主张

本课题的核心主张可以压缩为三条：

1. **问题主张**：推理图优化中的 rewrite 选择，本质上是一个图级决策问题，而不是局部启发式问题。
2. **方法主张**：候选 plan 的性能差异来源于结构相互作用，因此显式建模图结构的学习器应优于局部代价和非结构化特征模型。
3. **评估主张**：评价重点不应只是 latency prediction accuracy，而应是 fixed-budget 下的 plan selection quality，即在有限候选测试预算下能否选到更快的计划。

## 4. 研究问题定义

给定：

- 原始计算图 `G`
- 一组语义等价的候选 rewrite plans `P = {p1, p2, ..., pn}`
- 执行条件 `c`，包括硬件、precision、batch size、sequence length 等

学习一个评分函数：

`f(G, pi, c) -> score`

使得该函数能够根据候选 plan 的图结构与执行条件，对 plan 排序，并选择在真实执行中 latency 最优或接近最优的计划。

更具体地说，可以把预测目标设计为以下三种之一：

1. 绝对 latency regression
2. 相对 latency delta prediction
3. pairwise / listwise ranking

当前最推荐的主线是：

- 主目标：**pairwise ranking**
- 辅助目标：**delta-latency regression**

因为最终任务是选 plan，而不是报毫秒数。

## 5. 方法主线

### 5.1 图表示

每个候选 rewrite plan 对应一个候选图，或候选图在 rewrite 作用域扩张后的局部子图。建议使用有向异构图表示，包含：

#### 节点特征

- op type
- tensor shape
- dtype
- layout
- estimated FLOPs
- estimated memory bytes
- 是否为 fusion/rewrite 关键节点
- 所处拓扑层级或 block 内位置

#### 边特征

- data dependency
- control/order dependency
- tensor size proxy
- layout-sensitive edge

#### 全局条件特征

- GPU 型号
- precision 模式
- batch size
- sequence length
- 静态编译配置

### 5.2 模型

推荐从较稳妥的图模型开始，而不是一开始追求复杂架构：

- heterogenous GNN
- DAG-aware message passing network
- graph readout + MLP scorer

重点不是模型花哨，而是：

1. 能编码 rewrite 后的结构差异
2. 能融合全局执行条件
3. 能输出 plan-level score

### 5.3 学习目标

建议主训练目标为：

- pairwise ranking loss

辅助目标可加入：

- delta-latency regression loss

这样做的原因是：

- ranking 更贴近 plan selection 任务本质
- regression 可以提供更平滑的优化信号
- 两者结合有利于提升排序稳定性

## 6. 系统中的插入位置

本方法属于 **推理系统的编译优化层**，更准确地说，位于：

> **图优化器中的 rewrite-plan ranking / cost-model 层，处于候选改写枚举之后、lowering 与 autotuning 之前。**

它负责：

1. 对候选 rewrite plans 进行快速打分
2. 剪掉低潜力候选
3. 把少量高潜力候选送去真实编译或 profile

因此，它不是 runtime scheduler，也不是 kernel implementation，而是一个学习驱动的 plan selector。

## 7. 与“大模型推理服务”的关系

这个课题**仍然属于大模型推理服务范畴，但更准确地说，它位于大模型推理服务系统栈中的离线图优化/编译层，而不是在线请求调度层**。

也就是说：

- 如果从系统边界看，它服务于 LLM inference serving
- 如果从方法落点看，它修改的是 serving 系统底层的 graph optimization / compilation pipeline

因此它不是传统意义上的“请求调度、KV cache 管理、continuous batching”问题，但仍然可以被合理归入“面向大模型推理服务性能优化的方法”。

更稳妥的表达是：

> 面向大模型推理执行图优化的编译层学习方法。

## 8. 最小可行研究范围

为保证十月前形成 ICLR 可投稿工作，必须严格控制范围。

### 8.1 推荐范围

- 单卡 GPU
- 单一 backend
- 离线 profile
- 固定 1–2 类高频 block

优先块类型：

- Transformer MLP block
- Attention 相关 block

### 8.2 推荐 rewrite family

第一阶段建议只做以下 2–3 类：

- fusion-related rewrites
- transpose/layout movement
- reassociation / elementwise reorder

### 8.3 明确不做

第一阶段不要引入：

- 多 backend 联合建模
- 全模型任意 rewrite
- 大规模 e-graph 全空间搜索
- 在线 runtime 集成
- 多机多卡实验

## 9. 数据构造方案

### 9.1 数据来源

从开源 LLM 模型中提取高频 block，例如：

- LLaMA 系列 block
- Qwen 系列 block
- Mistral 系列 block

转换为统一中间表示，例如：

- ONNX
- Torch FX
- Relay

### 9.2 样本构造

对每个 block：

1. 枚举有限个合法 rewrite plans
2. 对每个候选 plan 进行真实编译与 profile
3. 记录固定条件下的稳态 latency
4. 形成 `(原图, 候选plan图, 条件, latency/ranking)` 数据

### 9.3 标签设计

可生成两类标签：

- pairwise preference label
- delta-latency label

同时保留 oracle best plan 作为评估基准。

## 10. Baseline 设计

至少应包含以下基线：

### 10.1 规则式启发式

- 优先 fusion 更多节点
- 优先减少 transpose
- 优先减少 estimated bytes

### 10.2 局部代价基线

- op-wise latency sum
- FLOPs + bytes analytical proxy

### 10.3 非图学习基线

- graph summary + MLP
- handcrafted features + XGBoost

### 10.4 非图深模型基线

- linearized graph + Transformer/MLP

### 10.5 Oracle

- 真实 profile 下的最优 plan

## 11. 实验设计

### 11.1 主要实验问题

1. 图级学习是否优于局部启发式？
2. GNN 是否优于非结构化学习基线？
3. 在固定搜索预算下，是否能选到更快的 plan？
4. 方法是否能在不同 block、不同 shape 上保持泛化？

### 11.2 核心指标

#### 预测/排序质量

- pairwise accuracy
- Spearman correlation
- Kendall tau
- top-k hit rate

#### 决策质量

- top-1 plan selection accuracy
- regret relative to oracle
- fixed-budget selected-plan latency
- search cost reduction

真正关键的是：

> **在相同搜索预算下，是否能选到更快的 rewrite plan。**

### 11.3 泛化实验

建议至少验证以下泛化维度：

- seen block -> unseen block
- seen shape -> unseen shape
- 不同 batch size / sequence length

## 12. 可能贡献

如果课题进展顺利，最终论文可主张的贡献应围绕以下三点组织：

1. 将推理图优化中的 rewrite 选择形式化为一个**等价计算图空间上的学习决策问题**。
2. 提出一个用于 rewrite plan ranking 的 **graph-level execution estimator**。
3. 在真实推理 block 的 rewrite 数据上证明：图级结构建模能够在有限搜索预算下选出更优计划。

## 13. 主要风险

### 风险 1：图模型不明显优于简单模型

可能原因：
- rewrite family 太简单
- 图级交互不够强
- 特征工程不足

应对：
- 优先选择 graph interaction 明显的 rewrite family
- 设计强非图 baseline，避免假优势

### 风险 2：标签噪声大

可能原因：
- warmup 不充分
- autotuning 干扰
- 异步执行与 profile 抖动

应对：
- 固定环境
- 多次测量取中位数
- 控制 workspace 与执行配置

### 风险 3：前端 rewrite 差异被后端抹平

可能原因：
- backend lowering 把不同候选映射成相同执行计划

应对：
- 优先选择 downstream 执行差异明显的 rewrite family
- 在样本构造阶段先验证 candidate diversity 是否真实存在

### 风险 4：问题被审稿人认为过于工程化

应对：
- 强调“equivalent graph space”上的学习问题
- 强化泛化实验与 decision-centric evaluation
- 避免把贡献写成某个系统的局部 patch

## 14. 近期执行路线

### 第一阶段：问题收缩与可行性验证

1. 确定 backend
2. 确定 block 类型
3. 确定 rewrite family
4. 写出候选 plan 枚举器
5. 验证是否存在显著的 plan-level latency 差异

### 第二阶段：数据集与基线

1. 建立 profile 流水线
2. 采集候选 plan 数据
3. 跑 heuristic / analytical / MLP baseline
4. 确认问题不是“简单模型已经做完”

### 第三阶段：图模型与主实验

1. 实现 GNN ranking 模型
2. 跑 pairwise ranking
3. 做 fixed-budget plan selection 实验
4. 做泛化与消融

### 第四阶段：论文收敛

1. 收紧 claim
2. 统一写作主线
3. 打磨图表
4. 准备 ICLR 投稿版本

## 15. 当前判断

这是一个适合作为当前主课题推进的方向，原因在于：

1. 它的问题定义足够硬，不只是工程 patch。
2. 它允许单卡离线闭环，适合当前阶段快速推进。
3. 它可以被包装成 AI 会可接受的“结构化学习决策”问题。
4. 它与大模型推理服务密切相关，但落点明确在编译优化层，边界清楚。

后续所有方案设计，都应优先服务于下面这件事：

> 尽快证明：在等价 rewrite plans 之间，图级学习方法能够在有限搜索预算下稳定选到更优计划。