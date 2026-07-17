# Rewrite 课题：验证后的两周执行清单与阶段路线

## 0. 结论先行

当前这条推进顺序总体是成立的：

> 范围锁定 → candidate plan 枚举 → profile 流水线 → 简单 baseline → 再决定是否上图模型

但经过检索核查后，需要做两点修正：

1. **简单 baseline 要更早介入**，不要等所有数据都铺开后再做。只要 profile 流水线能跑通，就应立刻做 1–2 个最弱 baseline。
2. **最终评价重点应放在 selected-plan latency / regret / fixed-budget selection quality**，而不是只看 latency regression 误差或 rank 指标本身。

整体判断：
- **确认成立**：研究顺序合理，符合 learned cost model / candidate selection 类工作的常见推进逻辑。
- **需要修正**：要更早检查 simple baseline gap，避免后期才发现问题过于简单。
- **风险提醒**：最大风险不是模型，而是 candidate diversity 不足、profile 噪声过大、以及 simple baseline 已经接近上限。

---

## 1. 已验证的判断

### 1.1 两周执行主顺序是合理的

已验证结论：
- 先做搜索空间/候选空间，再做测量，再做 cost model 或 learned selector，这个顺序是合理的。
- 对于 rewrite plan selection 这种问题，前期最重要的是先确认：
  - 候选 plan 是否真实存在
  - 候选之间是否有可分的 latency 差异
  - 测量标签是否稳定
  - 简单方法是否已经做得差不多了

证据强度：
- **直接支持**：TASO、Ansor、MetaSchedule 一类工作都把“候选空间 + 测量/调优 + 选择”放在前面，而不是先上复杂模型。
- **间接支持**：Bao、Balsa、Lero 这类 learned selection / learning-to-rank 工作都强调：模型是为决策服务，不应脱离候选空间与真实反馈单独讨论。

### 1.2 先验证 candidate diversity 和 profile stability 是必要的

已验证结论：
- 这一步非常合理，而且对你这个题几乎是前置门槛。
- 如果没有 candidate diversity，学习器没有东西可选。
- 如果 profile 不稳定，训练标签本身就不可信。

证据强度：
- **直接支持**：PyTorch benchmark 官方文档明确强调 warmup、同步、重复测量。
- **间接支持**：TVM MetaSchedule / Ansor 的实践都依赖高质量 measurement record；这本质上要求 profiling 信号稳定。
- **证据不足处**：没有发现单篇论文把“先做 candidate diversity + profile stability”写成严格定理；这更像强系统经验和高可信研究策略。

### 1.3 决策型评估比纯回归误差更合理

已验证结论：
- 你的题是 plan selection，不是单纯 latency prediction。
- 因此更合理的主指标应是：
  - top-1 selection accuracy
  - regret relative to oracle
  - fixed-budget selected-plan latency
  - top-k hit rate
- rank correlation、pairwise accuracy 可以保留，但更适合作为辅助指标。

证据强度：
- **直接支持**：Lero 明确采用 learning-to-rank 思路来服务最终选择。
- **间接支持**：Bao、Ansor、MetaSchedule 本质上都关注有限预算下选得更好，而不是纯 MSE 最小。

---

## 2. 验证后调整的两周执行清单

下面是修正后的版本。核心逻辑不变，但把 baseline 提前，并把每周的“止损检查点”明确化。

## 第 1 周：证明问题存在且可测

### Day 1：锁定最小实验边界
必须定死：
- 1 个 backend
- 1 种 IR
- 1 类 block
- 1 类 rewrite family

建议优先：
- block：`Transformer MLP block`
- rewrite family：`fusion-related` 或 `elementwise reorder`

当天产出：
- 一页最小实验设定
- 明确“不做什么”的边界清单

验收标准：
- 范围能在两周内闭环
- 不依赖多 backend、多机、多卡、完整 serving 集成

### Day 2：抽取 block 并统一表示
要做：
- 抽出一个最小 block
- 转为统一图表示
- 确认至少保留：op type、shape、dependency、关键 tensor 信息

当天产出：
- 1 个可重复提取的 block 样本
- 1 份图表示样例
- 1 个最小数据结构定义

### Day 3：定义 rewrite 空间
要做：
- 先定义一类有限、合法的 rewrite family
- 对单个 block 至少能构造出 `5~20` 个 candidate plans
- 确认这些 candidate 在结构上不是完全重复的

当天产出：
- rewrite rule 列表
- candidate plan 数据结构
- 单个 block 的候选样例

### Day 4：实现 candidate plan 枚举器
要做：
- 输入一个 block
- 程序化输出一批 candidate plans
- 每个 candidate 都能落成可执行图表示

当天产出：
- 最小可跑枚举器
- 至少 3 个 block 的 candidate 输出
- candidate 数量和去重统计

### Day 5：接 profile 流水线
要做：
- 对 candidate 做真实运行测量
- 固定 warmup、repeat、同步方式、执行条件
- 记录中位数、方差、winner 是否翻转

当天产出：
- 第一版 profile 脚本
- 单个 block 的 candidate latency 表

### Day 6：提前插入最弱 baseline
这是验证后新增强调的一步。

只要 Day 5 能跑通，就立刻做最简单 baseline：
- fusion 数量优先
- transpose 更少优先
- estimated bytes 更少优先
- FLOPs 更少优先

当天产出：
- heuristic baseline 排序结果
- 与 oracle 的初步对比

为什么这一步要提前：
- 越早知道“简单规则能不能做完”，越能判断题的真实难度。

### Day 7：做可行性判断
检查三件事：
- candidate diversity 是否足够
- profile 是否稳定
- simple heuristic 是否已经非常强

第 1 周结束时必须能回答：
- 这个 selection 问题在当前设定下是否真实存在？
- 当前 rewrite family 是否值得继续？
- 题是否过于简单或标签是否过于噪声？

止损标准：
- 如果 candidate 太少或差异太小，优先换 rewrite family
- 如果测量方差接近候选差异，先改 profile protocol，不要急着上模型

---

## 第 2 周：证明简单方法做不完，再决定是否上结构化模型

### Day 8：正式定义任务与标签
建议：
- 主任务：`pairwise ranking`
- 辅助任务：`delta-latency regression`

要做：
- 明确 pairwise label 生成规则
- 明确 oracle best plan 定义
- 明确 train/val/test 划分原则

当天产出：
- 任务定义文档
- 标签生成规则

### Day 9：做 analytical baseline
要做：
- op-wise latency sum
- FLOPs + bytes proxy
- graph summary features

当天产出：
- analytical baseline 结果表

### Day 10：做非图学习 baseline
要做：
- handcrafted features + XGBoost
- graph summary + MLP

当天产出：
- 至少 2 个简单学习 baseline
- 与 heuristic / oracle 的对比

### Day 11：做第一版统一评估
重点看：
- top-1 selection accuracy
- regret relative to oracle
- fixed-budget selected-plan latency
- top-k hit rate
- pairwise accuracy / rank correlation（辅助）

当天产出：
- 第一版总表
- 简单 baseline gap 结论

### Day 12：做“是否上图模型”的明确决策
分两种情况：

#### 情况 A：simple baseline 已接近上限
说明：
- 当前问题可能太简单
- 或当前 rewrite family 不够体现图级交互

此时优先动作：
- 换更难的 rewrite family
- 增加更复杂的上下文依赖
- 收缩 claim，不要硬上重模型

#### 情况 B：simple baseline 明显不够
说明：
- 图结构信息可能有增量价值
- 可以进入结构化模型阶段

当天产出：
- 是否值得上图模型的判断结论

### Day 13：设计第一版结构化模型
不建议直接上重型 GNN。
建议顺序：
1. graph summary + stronger MLP
2. linearized graph encoder
3. 轻量 GNN / DAG-aware message passing

当天产出：
- 输入特征表
- 模型输入输出格式
- ranking loss 定义

### Day 14：形成阶段性结论
必须整理出：
- 当前最值得保留的 block
- 当前最值得保留的 rewrite family
- profile 流水线是否稳定
- simple baseline 是否做不完
- 图模型是否值得上
- 下一阶段到底是扩数据还是进模型

最终产出：
- 一页研究判断总结

---

## 3. 验证后的阶段路线图

下面这份比“两周清单”更长，是整个课题从现在到投稿前更合适的阶段划分。

## 阶段 1：任务存在性与测量可行性验证

目标：
- 证明这个题不是空问题，也不是纯噪声问题。

要做：
- 收缩问题边界
- 构造 candidate plan 空间
- 跑小规模 profiling
- 提前跑最弱 baseline

过关条件：
1. 存在足够 candidate diversity
2. latency 测量有基本稳定性
3. simple baseline 有信息量但做不完

止损点：
- 候选空间太贫乏
- winner 经常翻转
- heuristic 几乎直接等于 oracle

如果失败：
- 优先换 rewrite family，而不是换模型

## 阶段 2：数据集、profile protocol 与 baseline 建设

目标：
- 把问题从“单点可行”变成“系统可评估”。

要做：
- 扩充 block 样本
- 固化 profiling 协议
- 固化 label policy
- 建完整 baseline 套件

过关条件：
1. 数据覆盖不再只靠少数样本
2. profiling protocol 固定可复现
3. baseline 套件完整且有明确 gap

止损点：
- 数据工程工作量快速失控
- 标签策略反复摇摆
- 不同 block 间完全无法形成统一任务

如果失败：
- 进一步缩 block 范围，先保住单一子问题闭环

## 阶段 3：结构化模型与主实验

目标：
- 证明结构化模型相对简单方法确有必要性。

要做：
- 设计轻量结构化输入
- 做 ranking 模型
- 跑主实验
- 比较不同模型家族

过关条件：
1. 结构化模型稳定优于强简单基线
2. 关键指标提升体现在 selection quality 上
3. 泛化设定下仍有优势

止损点：
- 模型只在极少数 workload 偶然赢
- gain 很小但复杂度很高
- 论文主线变成“模型堆砌”

如果失败：
- 改为更强 feature model 或缩 claim，不要硬保 GNN

## 阶段 4：泛化、消融与论文收敛

目标：
- 把结果整理成 AI 会可接受的完整论证。

要做：
- 泛化实验
- 消融实验
- failure case 分析
- 图表与论文主线统一

过关条件：
1. 能解释为什么方法有效
2. 能说明在哪些情况下会失效
3. 能形成清楚的 claim，而不是系统 patch 叙事

止损点：
- 泛化实验做太散
- 消融过多但主结论不清
- 写作时仍然像工程汇报而不是研究问题

如果失败：
- 收紧贡献表达，只保留最有说服力的一条主线

---

## 4. 现在最该盯住的几个决定点

在你真正进入模型阶段前，必须先明确这 5 个判断：

1. **候选 plan 是否足够丰富**
- 如果候选空间本身很弱，学习选择价值就会塌。

2. **profile 是否足够稳定**
- 如果标签不稳，后续模型大概率是在拟合噪声。

3. **simple baseline 是否已经太强**
- 如果简单方法已经接近 oracle，图模型的新意会很危险。

4. **任务是否应以 ranking 为主**
- 当前证据支持：是。

5. **是否值得上图模型**
- 不能先验假设值得，必须以前面 4 条为依据。

---

## 5. 当前最推荐的推进原则

如果只保留最重要的执行原则，就是下面 4 句：

1. 先证明候选可选，不要先证明模型多强。
2. 先证明标签可信，不要先堆复杂结构。
3. 先证明简单方法做不完，再谈图模型必要性。
4. 先把题做成离线研究闭环，再考虑接入 `chitu`。

---

## 6. 核查来源

以下来源用于支持上面的“顺序合理性、测量先行、决策型评估、分阶段推进”判断。注意：它们并不是都在直接研究 rewrite plan selection；其中一部分是**直接支持**，另一部分是**相邻方向的间接支持**。

### 直接或较直接相关
- [Optimizing Deep Learning Computation Graphs with Automated Generation of Graph Substitutions (TASO)](https://dl.acm.org/doi/10.1145/3341301.3359630)
- [Ansor: Generating High-Performance Tensor Programs for Deep Learning](https://www.usenix.org/conference/osdi20/presentation/zheng)
- [Apache TVM MetaSchedule Documentation](https://tvm.apache.org/docs/arch/meta_schedule.html)

### 间接支持：有限预算选择、学习排序、分阶段决策
- [Lero: A Learning-to-Rank Query Optimizer for Database Systems](https://www.vldb.org/pvldb/vol15/p3537-yang.pdf)
- [Bao: Learning to Steer Query Optimizers](https://arxiv.org/abs/2004.03814)
- [Balsa: Learning a Query Optimizer Without Expert Demonstrations](https://arxiv.org/abs/2201.01441)

### 测量稳定性与 profiling protocol
- [PyTorch Benchmark Documentation](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)

---

## 7. 最后判断

目前最稳的路线仍然是：

> 先把这个课题做成一个小而硬的离线 research loop，先验证 candidate diversity、profile stability、simple baseline gap；只有这些都成立后，图模型和后续 `chitu` 接入才值得投入。

也就是说，接下来最关键的不是“怎么尽快上 GNN”，而是：

> **怎么尽快证明这个 selection 问题真实存在，而且简单方法做不完。**
