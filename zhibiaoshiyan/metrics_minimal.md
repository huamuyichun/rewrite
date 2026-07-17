# Rewrite 课题最小指标说明

本文档给出当前课题 `Graph-level latency estimator for rewrite plan selection` 的保守版最小指标集。目标是：
- 能和工程师对齐
- 能稳定测量
- 能支撑论文中的“提速”叙事
- 不强行承诺难以预估的数值目标

## 1. 指标原则

本课题的核心不是做一个纯 latency regression 模型，而是做 rewrite plan selection。
因此指标应优先反映：
1. 最终是否更快
2. 是否选得更接近最优
3. 选择本身是否有额外开销
4. 结果是否稳定

不建议把预测误差作为主指标。

## 2. 必选指标

### 2.1 最终提速指标

**指标名**：Selected-plan latency speedup

**定义**：
- `speedup = latency(baseline_plan) / latency(selected_plan)`

**说明**：
- 这是最核心的结果指标。
- 反映方法最终是否真的让执行更快。
- 论文主文中应优先报告该指标。

**推荐汇报方式**：
- `median speedup`
- `geomean speedup`
- `win rate`

**保守说明**：
- 不预设具体数值目标。
- 只要求相对于 baseline 有稳定正收益。

### 2.2 选择质量指标

**指标名**：Oracle gap / regret

**定义**：
- `regret = latency(selected_plan) / latency(best_plan_in_candidate_set) - 1`

**说明**：
- 衡量所选 plan 距离候选集合内最优 plan 还有多远。
- 这是 plan selection 问题的核心评价之一。

**推荐汇报方式**：
- 平均 regret
- 中位数 regret
- 分 block 的 regret 分布

**保守说明**：
- oracle 仅定义为“当前候选集合内的最优实测 plan”，不承诺全局最优。

### 2.3 选择开销指标

**指标名**：Decision overhead

**定义**：
- `decision time per graph`
- 如系统需要筛候选，则补充 `#measured candidates per graph`

**说明**：
- 反映为了做出选择额外花了多少时间。
- 防止方法在工程上“收益被选择成本吃掉”。

**推荐汇报方式**：
- 平均决策时间
- 中位数决策时间
- 候选测量数量

**保守说明**：
- 不预设具体时延阈值。
- 只要求选择开销不成为主要瓶颈。

### 2.4 稳定性指标

**指标名**：Win rate / repeatability

**定义**：
- `win rate`：在多少比例的图上优于 baseline
- `repeatability`：少量重复测量下的方差或排序一致性

**说明**：
- 用来确认结果不是少数样本拉高的假平均值。
- 也用于验证 profiling protocol 是否可信。

**推荐汇报方式**：
- win rate
- 少量样本的 repeated-run variance

**保守说明**：
- 重复测量只做抽样，不要求全量重复。

## 3. 可选指标

这些指标可作为辅助分析，但不作为当前必选项：

- `top-k hit rate`
- `pairwise accuracy`
- `Spearman / Kendall`
- `MAE / RMSE`
- `budget-performance curve`

说明：
- 这些指标有助于分析模型行为。
- 但它们不如 speedup / regret / overhead 更贴近最终目标。

## 4. 不建议先承诺的指标

以下指标不建议作为当前阶段的硬承诺：

- 全局最优发现率
- 真实线上端到端收益
- 多硬件全面泛化
- p99 / p999 尾延迟主指标
- 以 RMSE 作为主结论指标

原因：
- 这些指标要么过难预估，要么对当前阶段来说风险过高。
- 它们会显著增加实验复杂度，并可能让前期验证失焦。

## 5. 最小指标集建议

如果只保留最保守、最容易和工程师对齐的一组指标，建议如下：

1. `Selected-plan latency speedup`
2. `Oracle gap / regret`
3. `Decision overhead`
4. `Win rate`

这是当前阶段最推荐的最小指标集。

## 6. 对工程对接的口径建议

可以直接和工程师这样对齐：

- 我们先只定义四类最小指标：最终提速、选择质量、选择开销、稳定性。
- 数值目标不先硬承诺，先保证这四类指标能稳定测出来。
- 其中最核心的是 selected-plan latency speedup，其次是 regret 和 decision overhead。
- 预测误差类指标只作为辅助，不作为项目主验收标准。

## 7. 当前版本结论

当前阶段最保守、最稳妥的指标方案是：

- 主指标：`Selected-plan latency speedup`
- 辅助核心指标：`Oracle gap / regret`
- 工程约束指标：`Decision overhead`
- 稳健性指标：`Win rate`

如果后续实验顺利，再考虑加入 top-k、rank correlation、budget curve 等扩展指标。
