# Day 4 日志

日期：2026-05-26

## 1. 做了什么

创建 Day 4 工作目录：

```text
/pub/data/hjwz/rewrite/vertify/day4
```

创建脚本：

```text
instantiate_candidates.py
```

该脚本读取 Day 3 的 `candidate_plans.json`，把 6 个 candidate plans 实例化为可执行 PyTorch modules，并为每个 candidate 抽取 Torch FX graph。

## 2. 目的是什么

Day 4 的目的不是测 latency，而是确认：

```text
Day 3 定义的候选 rewrite plans 是否真正可执行、可抽图、结构不重复、并且与 baseline 数值等价。
```

如果 Day 4 失败，Day 5 profiling 没有意义。

## 3. 当前候选

当前处理 6 个 candidates：

```text
p0_baseline_separate_silu
p1_separate_manual_silu
p2_separate_inplace_silu_mul
p3_fused_chunk_silu
p4_fused_split_silu_inplace
p5_fused_chunk_manual_silu
```

## 4. 当前边界

Day 4 不做：

- latency profiling；
- heuristic baseline；
- oracle selection；
- torch.compile timing；
- GNN；
- chitu 接入。

## 5. 预期输出

脚本运行后应生成：

```text
candidate_summary.csv
candidate_summary.json
equivalence_results.csv
equivalence_results.json
dedup_result.json
metadata.json
candidate_graphs/
```

## 6. 验收标准

Day 4 通过条件：

1. 6 个 candidates 均可实例化；
2. 6 个 candidates 均可执行；
3. 6 个 candidates 均可 trace 成 FX graph；
4. 6 个 candidates 与 baseline 数值等价；
5. 实际 FX graph 结构签名无重复。

## 7. 单 block 实际运行结果

运行：

```text
conda activate rewrite_miniexp
python /pub/data/hjwz/rewrite/vertify/day4/instantiate_candidates.py --out-dir /pub/data/hjwz/rewrite/vertify/day4
```

结果：

```text
torch = 2.10.0+cu129
device = NVIDIA A100 80GB PCIe
num_candidates = 6
num_unique_actual_signatures = 6
dedup_status = ok
equivalence_status = ok
num_equivalence_passed = 6
atol = 0.002
rtol = 0.02
```

说明：

- 6 个 Day 3 candidates 均可实例化；
- 6 个 candidates 均可执行 forward；
- 6 个 candidates 均可 trace 成 FX graph；
- 6 个 candidates 与 baseline 在 tolerance 下等价；
- 实际 FX graph 结构签名无重复。

## 8. 多 block 输出结果

为满足 roadmap 中 Day 4 的要求，额外定义 3 个 block specs：

```text
b0_seq128_h1024_i4096
b1_seq128_h768_i3072
b2_seq512_h768_i3072
```

运行：

```text
python /pub/data/hjwz/rewrite/vertify/day4/run_block_batch.py
```

汇总结果：

```text
status = ok
num_blocks = 3
num_total_candidate_outputs = 18
errors = []
```

每个 block 均满足：

```text
num_candidates = 6
dedup_status = ok
num_unique_actual_signatures = 6
equivalence_status = ok
num_equivalence_passed = 6
```

输出文件：

```text
block_specs.json
block_candidate_summary.csv
block_run_summary.json
block_outputs/
```

## 9. Day 4 结论

Day 4 验收通过。

已经完成：

1. 最小可跑枚举器雏形：`instantiate_candidates.py` 可根据 candidate plan 定义生成可执行 candidate；
2. 至少 3 个 block 的 candidate 输出：3 个 block，共 18 个 candidate outputs；
3. candidate 数量和去重统计：每个 block 6 个 candidate，实际 FX 结构签名均不重复；
4. 数值等价检查：所有 candidate 均通过 baseline equivalence check。

下一步 Day 5 可以开始 profiling 流水线，对这些 candidates 做真实 latency 测量。
