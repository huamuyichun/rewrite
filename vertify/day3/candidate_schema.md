# Day 3 Candidate Plan 数据结构

日期：2026-05-25

## 1. 目的

本文档定义 Day 3 的 candidate plan 最小数据结构。

当前只定义 candidate，不测 latency，不生成 ranking label。

## 2. CandidatePlan 字段

`candidate_plans.json` 中每个 candidate 使用以下字段：

| 字段 | 含义 |
| --- | --- |
| `candidate_id` | candidate 唯一 ID |
| `is_baseline` | 是否为默认 baseline plan |
| `rewrite_family` | 所属 rewrite family |
| `plan_desc` | 人类可读描述 |
| `gate_up_projection` | `separate` 或 `fused` |
| `gate_up_split` | `none`、`chunk` 或 `split` |
| `activation` | `f_silu` 或 `manual_silu` |
| `multiply` | `out_of_place` 或 `inplace` |
| `expected_fx_pattern` | 预期 FX 图结构 |
| `semantic_condition` | 合法性/语义等价条件 |
| `structural_signature` | 用于检查结构重复的签名 |

## 3. 与 Day 2 数据结构的关系

Day 2 定义的是原始 block 图：

```text
G = (V, E, X)
```

Day 3 在此基础上定义：

```text
CandidatePlan = (candidate_id, rewrite_rule_choices, semantic_condition, expected_graph_signature)
```

后续 Day 4 才会把每个 `CandidatePlan` 实例化为：

```text
candidate_fx_graph
candidate_module
equivalence_check_result
dedup_signature
```

## 4. 当前不包含的字段

Day 3 不包含：

- measured latency；
- oracle label；
- pairwise label；
- selector score；
- profiling run id；
- compile artifact。

这些字段属于 Day 5 以后。

## 5. 最小合法性要求

每个 candidate 必须满足：

1. 来自同一个 block；
2. 属于同一个 rewrite family；
3. 与 baseline 语义等价或可通过 Day 4 的数值等价检查验证；
4. 有明确结构签名；
5. 可以在 Day 4 中落成可执行 PyTorch module / FX graph。
