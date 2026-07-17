# Day 1 日志

日期：2026-05-25

## 1. 做了什么

创建 Day 1 工作目录：

```text
/pub/data/hjwz/rewrite/vertify/day1
```

在该目录下创建：

- `README.md`：Day 1 总览、目的、结论、验收；
- `day1_scope.md`：前两周验证任务的最小实验边界；
- `LOG.md`：本日志，记录做了什么、目的是什么、结论是什么。

同时确认了一个路径问题：

- 用户提到的目录名是 `vertify`；
- 当前 `/pub/data/hjwz/rewrite` 下已有目录是 `verity`；
- 本次按用户最新指令新建并使用 `vertify/day1`；
- 未删除、未移动、未改动已有 `verity`。

## 2. 目的是什么

Day 1 的目的不是实验执行，而是范围锁定。

前两周的总目标是验证：

1. rewrite candidate plans 是否有稳定 latency 差异；
2. baseline/default plan 距离 oracle plan 是否存在可利用 gap；
3. simple heuristic 是否不能完全解决问题；
4. selector decision overhead 是否足够低。

Day 1 只完成其中的前置动作：

```text
把实验对象、backend、IR、rewrite family 和不做项固定下来。
```

## 3. 为什么这样做

如果 Day 1 不先固定边界，后续很容易扩散到：

- attention；
- 多 backend；
- chitu 深度接入；
- e-graph；
- GNN 模型；
- 大规模 shape 泛化。

这些都不是前两周最该做的事。

当前最重要的问题是：

```text
这个 rewrite plan selection 问题是否真实存在，而且简单方法是否做不完。
```

所以 Day 1 必须先把问题压缩到最小可验证版本。

## 4. 固定结论

Day 1 固定如下：

```text
backend: PyTorch eager + torch.compile(inductor)
IR: Torch FX
block: Transformer MLP block
rewrite family: Linear/GEMM 后逐点链的等价重排，优先 fusion-related gate/up projection variants
GPU: 单卡
dtype: fp16 优先
batch_size: 1
shape: 先单 shape 跑通，再扩展到 seq_len 128 / 512 / 2048
```

第一阶段不做：

```text
attention 主实验
多 backend
多机多卡
全模型 rewrite
e-graph 全空间搜索
online serving 集成
chitu 深度接入
复杂 GNN
大规模 shape 泛化
```

## 5. 当前结论

Day 1 已完成。

当前项目应继续按下面顺序推进：

```text
Day 2: 抽取最小 Transformer MLP block，并得到 Torch FX graph
Day 3: 定义有限 candidate rewrite plans
Day 4: 实现 candidate plan 枚举和去重
Day 5: 接 profiling 流水线
Day 6: 提前跑 simple heuristic baseline
Day 7: 判断问题是否存在、测量是否稳定、是否值得继续
```

关键判断：

- 不能直接上 GNN；
- 不能现在深接 chitu；
- 不能把指标主线写成 latency regression；
- 必须先验证 candidate diversity、profile stability 和 simple baseline gap。
