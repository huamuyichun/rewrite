# Day 4: Candidate Instantiation

日期：2026-05-26

## 目的

Day 4 的目标是把 Day 3 定义的 candidate plans 程序化落成可执行 PyTorch module 和 Torch FX graph。

本阶段回答：

```text
这些 candidate plans 是否真的能运行、能抽图、能去重、能通过数值等价检查？
```

本阶段不做：

- latency profiling；
- heuristic selector；
- oracle 计算；
- GNN；
- chitu 接入。

## 输入

Day 3 candidate 定义：

```text
/pub/data/hjwz/rewrite/vertify/day3/candidate_plans.json
```

## 输出

运行 `instantiate_candidates.py` 后生成：

- `candidate_summary.csv/json`：每个 candidate 的实际 FX 图摘要；
- `equivalence_results.csv/json`：每个 candidate 与 baseline 的数值等价检查；
- `dedup_result.json`：实际 FX 图结构签名去重结果；
- `metadata.json`：运行环境和状态；
- `candidate_graphs/<candidate_id>/`：每个 candidate 的 FX graph、节点表和实际结构签名；
- `LOG.md`：Day 4 日志。

## 验收标准

Day 4 通过条件：

1. 6 个 Day 3 candidates 都能实例化为 PyTorch module；
2. 6 个 candidates 都能被 Torch FX trace；
3. 6 个 candidates 都能执行 forward；
4. 6 个 candidates 与 baseline 输出在指定 tolerance 下等价；
5. 6 个 candidates 的实际 FX 结构签名不重复。

通过后才能进入 Day 5 profiling。
