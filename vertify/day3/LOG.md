# Day 3 日志

日期：2026-05-25

## 1. 做了什么

创建 Day 3 工作目录：

```text
/pub/data/hjwz/rewrite/vertify/day3
```

基于 Day 2 的 FX graph 定义了第一版 candidate rewrite space。

创建文件：

- `rewrite_rules.md`：rewrite family 和具体规则；
- `candidate_schema.md`：candidate plan 最小数据结构；
- `candidate_plans.json`：6 个 candidate plans；
- `validate_candidate_space.py`：candidate space 一致性检查脚本；
- `README.md`：Day 3 总览；
- `LOG.md`：本日志。

## 2. 目的是什么

Day 3 的目的不是跑实验，而是回答：

```text
在 Day 2 的 MLP FX graph 上，我们到底允许哪些语义等价 rewrite candidates？
```

如果没有这个定义，Day 4 的枚举器和 Day 5 的 profiling 都没有清楚输入。

## 3. 当前 Rewrite Family

第一阶段只保留一个 rewrite family：

```text
fusion_related_gate_up_projection
```

它围绕 SwiGLU-like MLP 中的 gate/up projection 和后续逐点链展开。

## 4. 当前 Candidate Set

当前固定 6 个 candidates：

```text
p0_baseline_separate_silu
p1_separate_manual_silu
p2_separate_inplace_silu_mul
p3_fused_chunk_silu
p4_fused_split_silu_inplace
p5_fused_chunk_manual_silu
```

其中：

- 1 个 baseline；
- 5 个非 baseline；
- 每个 candidate 有不同 structural signature；
- 每个 candidate 都记录了语义条件；
- 每个 candidate 后续都应能落成 PyTorch/FX 可执行图。

## 5. 关键限制

Day 3 只定义 candidate space。

尚未完成：

- 可执行 FX graph 生成；
- 数值等价检查；
- compile；
- latency profiling；
- heuristic selector。

这些属于 Day 4 以后。

## 6. 当前结论

Day 3 的核心产出已经完成：

1. rewrite rule 列表；
2. candidate plan 数据结构；
3. 单个 block 的 candidate 样例。

下一步 Day 4 应做：

```text
把这 6 个 candidate plans 程序化落成可执行 module / FX graph，并做结构去重和数值等价检查。
```

## 7. 一致性检查结果

运行：

```text
python /pub/data/hjwz/rewrite/vertify/day3/validate_candidate_space.py
```

检查结果：

```text
status = ok
num_candidates = 6
num_baselines = 1
rewrite_families = ["fusion_related_gate_up_projection"]
num_unique_candidate_ids = 6
num_unique_structural_signatures = 6
errors = []
```

结论：

```text
Day 3 candidate space 定义通过一致性检查。
```
