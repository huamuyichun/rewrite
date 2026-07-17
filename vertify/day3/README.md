# Day 3: Candidate Rewrite Space Definition

日期：2026-05-25

## 目的

Day 3 的目标是基于 Day 2 的 FX graph，定义一个有限、合法、可落地的 candidate rewrite space。

本阶段只做定义，不做 profiling。

## 输入

Day 2 已得到的 MLP FX graph：

```text
x -> gate_proj
x -> up_proj
gate_proj -> silu
silu + up_proj -> mul
mul -> down_proj
down_proj -> output
```

## 输出

Day 3 输出：

- `rewrite_rules.md`：rewrite family 和规则列表；
- `candidate_schema.md`：candidate plan 最小数据结构；
- `candidate_plans.json`：6 个候选 plan 样例；
- `validate_candidate_space.py`：检查 candidate 数量、baseline 数量、结构签名唯一性；
- `validation_result.json`：candidate space 检查结果；
- `LOG.md`：Day 3 日志。

## 当前 Candidate Set

当前固定 6 个 candidate plans：

1. `p0_baseline_separate_silu`
2. `p1_separate_manual_silu`
3. `p2_separate_inplace_silu_mul`
4. `p3_fused_chunk_silu`
5. `p4_fused_split_silu_inplace`
6. `p5_fused_chunk_manual_silu`

其中：

- 1 个 baseline；
- 5 个非 baseline；
- 1 个 rewrite family；
- 每个 candidate 有不同 structural signature。

## Day 3 边界

Day 3 不做：

- 把 candidate 真正转换成可执行 FX graph；
- 数值等价检查；
- latency profiling；
- heuristic baseline；
- GNN。

这些属于 Day 4 以后。
