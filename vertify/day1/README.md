# Day 1: Minimal Scope Lock

日期：2026-05-25

## 目的

Day 1 的目的不是跑大实验，也不是上 GNN，而是把接下来两周的验证问题收缩到一个能闭环的最小范围。

两周任务要验证的是：

1. candidate rewrite plans 之间是否存在稳定 latency 差异；
2. profiling label 是否可信，不是主要由测量噪声决定；
3. simple baseline 是否做不完，是否还给学习式 selector 留出空间。

Day 1 只负责完成第一步：固定实验边界。

## Day 1 结论

第一阶段实验边界固定为：

- backend：PyTorch eager + torch.compile(inductor)
- IR：Torch FX
- block：Transformer MLP block
- rewrite family：Linear/GEMM 后逐点链的等价重排，优先 fusion-related gate/up projection variants
- hardware scope：单卡 GPU
- dtype：优先 fp16；如果环境中 bf16 更稳定，再明确切换并记录原因
- batch size：1
- shape：先固定单 shape 跑通，再扩展到 seq_len 128 / 512 / 2048

第一阶段明确不做：

- 不做 attention 主实验
- 不做多 backend
- 不做多机多卡
- 不做全模型 end-to-end rewrite
- 不做 e-graph 全空间搜索
- 不做 online serving 集成
- 不做 chitu 深度接入
- 不做复杂 GNN
- 不做大规模 shape 泛化

## 为什么这样定

Transformer MLP block 是当前最稳的最小对象：结构简单、容易抽取、容易用 Torch FX 表示，也更容易构造少量语义等价的 candidate plans。

PyTorch + Torch FX + Inductor 的路线适合前两周验证问题是否存在。它比直接接 chitu 更快闭环，也比 ONNX / Relay 更少引入额外工程不确定性。

rewrite family 先选 Linear/GEMM 后的逐点链和 fusion-related 变体，是因为这类候选最容易形成“语义等价但执行表现不同”的 plan set，适合先验证 candidate diversity 和 profiling stability。

## Day 1 验收

Day 1 已完成：

1. 固定 backend；
2. 固定 IR；
3. 固定 block；
4. 固定 rewrite family；
5. 固定初始实验环境约束；
6. 写明第一阶段不做项；
7. 保留日志，记录目的、动作和结论。

下一步进入 Day 2：

抽取一个最小 Transformer MLP block，用 Torch FX 得到可重复的图表示，并定义最小 graph/candidate 数据结构。
