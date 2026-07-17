# Day1-Day5 研究决策诊断

- 结论：有条件继续
- 当前最大瓶颈：candidate space 太窄，oracle ceiling 不够高，且样本只有 3 个 block
- 当前最强证据：compile backend 下 3 个 block 都有非零 spread，b0/b1 spread 约 12.5% 且 winner 不翻转
- 当前最危险风险：b2 的 winner flip 说明部分 shape 上 profiling 噪声已经接近候选差异
- 下一步唯一最推荐动作：先扩 candidate space，并同步保留 Day5 profiling protocol 复测，不要直接上图模型

## 当前设定摘要

| 项目 | 当前值 | 证据文件 |
| --- | --- | --- |
| block | Transformer MLP / SwiGLU-like | day1_scope.md, day2/data_schema.md |
| IR | Torch FX | day2/fx_nodes.csv |
| rewrite family | fusion_related_gate_up_projection | day3/rewrite_rules.md |
| backend 主口径 | torch.compile(inductor) | day5/compile_reset/profile_run_summary.json |
| 补充 backend | PyTorch eager | day5/eager/* |
| device | NVIDIA A100 80GB PCIe | day5/compile_reset/profile_run_summary.json |
| dtype | fp16 | day4/block_specs.json |
| batch_size | 1 | day4/block_specs.json |
| seq_len | 128, 512；未覆盖 2048 | day4/block_specs.json |
| profile protocol | warmup=10, measure=30, CUDA Event + synchronize | day5/README.md, day5/LOG.md |

当前 FX graph 样例来自 `day2/fx_nodes.csv`：`x -> gate_proj / up_proj -> silu -> mul -> down_proj -> output`。当前 candidate set 来自 `day3/candidate_plans.json`，共 6 个候选，其中 1 个 baseline、5 个非 baseline。

## 关键统计表

### Core Stats

| 统计项 | 数值 | 解释 |
| --- | --- | --- |
| 每图 candidate 数量分布 | 6: 3 graph | 3 个 block 都是 6 个候选 |
| 每图去重后 candidate 数量分布 | 6: 3 graph | 实际 FX 结构没有重复 |
| spread median | 12.50% | best-vs-worst latency 差异中位数 |
| spread p90 | 12.56% | 样本只有 3 个，p90 仅供粗看 |
| candidate latency CV median | 4.14% | 18 个 candidate 的 CV 中位数 |
| candidate latency CV p90 | 5.22% | 18 个 candidate 的 CV p90 |
| block median CV median | 4.62% | 按 block 聚合后的 median CV |
| block p90 CV median | 5.32% | 按 block 聚合后的 p90 CV |
| winner flip | 1/3 | b2 发生 flip，b0/b1 未发生 |
| raw rows | 720 | warmup 180, measure 540；CSV 含 header 共 721 行 |

### Candidate Enumeration / Dedup

| block_id | seq | hidden | intermediate | candidate 去重前 | candidate 去重后 | equivalence | dedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| b0_seq128_h1024_i4096 | 128 | 1024 | 4096 | 6 | 6 | ok | ok |
| b1_seq128_h768_i3072 | 128 | 768 | 3072 | 6 | 6 | ok | ok |
| b2_seq512_h768_i3072 | 512 | 768 | 3072 | 6 | 6 | ok | ok |

### Compile Profiling By Block

| block_id | baseline p50 ms | best candidate | best p50 ms | oracle gain vs baseline | best-vs-worst spread | median CV | p90 CV | winner flip | 判断 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b0_seq128_h1024_i4096 | 0.096768 | p5_fused_chunk_manual_silu | 0.086016 | 12.50% | 12.50% | 4.62% | 5.32% | False | 较真实 |
| b1_seq128_h768_i3072 | 0.097280 | p3_fused_chunk_silu | 0.089600 | 8.57% | 12.57% | 4.73% | 5.88% | False | 较真实 |
| b2_seq512_h768_i3072 | 0.108544 | p3_fused_chunk_silu | 0.104448 | 3.92% | 3.92% | 2.88% | 3.11% | True | 偏弱/需复测 |

### Shape / Seq 对比

| seq_len | blocks | median spread | p90 spread | median p90 CV | winner flip rate | median oracle gain | 判断 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 2 | 12.54% | 12.56% | 5.60% | 0.00% | 10.54% | 更强 |
| 512 | 1 | 3.92% | 3.92% | 3.11% | 100.00% | 3.92% | 较弱/证据不足 |

### Candidate Rank 粗看

| candidate | avg rank | best rank | worst rank | 含义 |
| --- | --- | --- | --- | --- |
| p0_baseline_separate_silu | 5.67 | 5 | 6 | 弱/不稳定 |
| p1_separate_manual_silu | 4.00 | 4 | 4 | 弱/不稳定 |
| p2_separate_inplace_silu_mul | 5.33 | 5 | 6 | 弱/不稳定 |
| p3_fused_chunk_silu | 1.33 | 1 | 2 | 稳定强 |
| p4_fused_split_silu_inplace | 2.67 | 2 | 3 | 中等 |
| p5_fused_chunk_manual_silu | 2.00 | 1 | 3 | 稳定强 |

## Candidate Diversity 诊断

当前 candidate diversity 有两个层面：

1. **结构 diversity 成立**：Day4 的 `block_run_summary.json` 显示 3 个 block 均为 `num_candidates=6`、`num_unique_actual_signatures=6`、`dedup_status=ok`，说明 6 个候选在 FX 结构层面没有被去重抹平。
2. **性能 diversity 有但不够强**：Day5 compile 结果显示 spread 分别为 12.50%、12.57%、3.92%，median spread 为 12.50%，p90 spread 为 12.56%。

研究判断：candidate space 不是空的，但还不够强。b0/b1 有可用信号，b2 的 spread 太低。当前 6 个候选主要围绕 gate/up fusion、chunk/split、manual silu、inplace multiply，仍然太窄，不足以支撑“图模型必要性”的结论。

## Profiling Stability 诊断

Profiling protocol 是规范的：`warmup=10`、`measure=30`、CUDA Event 计时、前后 synchronize、compile prime time 单独记录、Inductor cache 放在 `/pub/data/hjwz/.cache/torchinductor`。

稳定性结论分 shape 看：

- b0：spread 12.50%，p90 CV 5.32%，winner 不翻转，信号可用。
- b1：spread 12.57%，p90 CV 5.88%，winner 不翻转，信号可用。
- b2：spread 3.92%，p90 CV 3.11%，winner 翻转，信号偏弱。

整体 18 个 candidate 的 CV median 为 4.14%，p90 为 5.22%。这个噪声水平对 b0/b1 尚可，对 b2 已经危险。

## 哪些结果说明“问题真实存在”

- 6 个 candidate 在每个 block 上都能实例化、trace、数值等价，且实际 FX signature 不重复。
- compile backend 下 3 个 block 都有 best-vs-worst spread，不是完全被 Inductor lowering 抹平。
- b0/b1 的 winner 不翻转，说明至少在 seq=128 的两个配置上，candidate latency ordering 有稳定信号。
- baseline 并非总是最优：b0/b1/b2 的 best candidate 都不是 baseline。

## 哪些结果说明“当前上限不够高”

- 样本数只有 3 个 block，无法支持泛化判断。
- oracle gain vs baseline 只有 12.50%、8.57%、3.92%；b2 的上限尤其低。
- 当前 rewrite family 只覆盖 gate/up fusion 周边，candidate 空间太窄。
- b2 的 winner flip 说明在低 spread 情况下，排序标签可能不稳定。
- 尚未跑 Day6 heuristic，不能证明 simple baseline 做不完。

## 研究判断问题回答

1. **当前 candidate space 是否足够强，还是 oracle ceiling 太低？**  
   有条件够用，但 oracle ceiling 偏低：6 个候选都不重复，spread median 12.50%，但 oracle gain vs baseline 只有 3.92%-12.50%，且样本数仅 3。

2. **当前 profiling 信号是否可信，还是噪声已经接近候选差异？**  
   部分可信：protocol 规范，b0/b1 信号大于噪声且 winner 稳；b2 spread 3.92%、p90 CV 3.11%、winner flip，噪声已接近候选差异。

3. **当前问题在哪些 shape 上最真实，哪些 shape 上最弱？**  
   seq=128 的 b0/b1 最真实；seq=512 的 b2 最弱。未覆盖 seq=2048，证据不足。

4. **现有 rewrite family 是否值得保留，还是应该优先换 family？**  
   保留但必须扩展：fusion_related_gate_up_projection 能产生可测差异，但上限不够高，不能只靠这 6 个候选进入图模型阶段。

5. **下一步投入排序**  
   1) 扩 candidate space；2) 分 shape/seq 做 heuristic；3) 上弱学习器；4) 暂不直接上图模型。

理由：当前还没有 Day6 heuristic 结果，直接上 MLP/XGBoost 或图模型都太早。最先要把 oracle ceiling 拉高，否则学习器没有足够可学空间。扩 candidate 后再按 shape/seq 做规则 baseline，确认 simple baseline gap。弱学习器应该在候选空间和标签稳定性更扎实后再上。图模型排最后，因为现在没有证据证明非图模型做不完。

## 失败样本分析

注意：当前只有 3 个 block，不足以列出真正的 Top 5 / Bottom 5。下面列出现有全部样本排序；“Top 5/Bottom 5”在当前数据下证据不足。

### 最有优化潜力的 graph / shape

| 排序 | block_id | spread | oracle gain | winner flip | 可能原因 |
| --- | --- | --- | --- | --- | --- |
| 1 | b1_seq128_h768_i3072 | 12.57% | 8.57% | False | fused gate/up 系列明显优于 separate；候选差异未被 Inductor 完全抹平 |
| 2 | b0_seq128_h1024_i4096 | 12.50% | 12.50% | False | fused gate/up 系列明显优于 separate；候选差异未被 Inductor 完全抹平 |
| 3 | b2_seq512_h768_i3072 | 3.92% | 3.92% | True | 有 spread 但 winner 不稳，噪声或候选差距接近 |

### 最没优化潜力的 graph / shape

| 排序 | block_id | spread | oracle gain | winner flip | 可能原因 |
| --- | --- | --- | --- | --- | --- |
| 1 | b2_seq512_h768_i3072 | 3.92% | 3.92% | True | 候选差异小且 winner flip；profiling 噪声接近候选差异 |
| 2 | b0_seq128_h1024_i4096 | 12.50% | 12.50% | False | 有一定差异，但当前候选上限仍不高 |
| 3 | b1_seq128_h768_i3072 | 12.57% | 8.57% | False | 有一定差异，但当前候选上限仍不高 |

## 下一步优先级建议

1. **扩 candidate space**  
   继续保留 gate/up fusion，但增加更能拉开 backend 差异的候选：layout/transpose movement、view/contiguous 周边、down_proj 前后的 elementwise regroup，或更明确影响 Inductor fusion 边界的 patterns。目标是让 median spread 稳定超过 p90 CV 的 2 倍，并让 oracle gain 不只停在 3%-8%。

2. **分 shape/seq 做 heuristic**  
   Day6 不应只做一个全局 heuristic。至少按 seq=128 和 seq=512 分开看，因为当前 b0/b1 与 b2 的信号强度不同。需要比较 baseline、fusion-prior heuristic、manual-silu-prior heuristic、inplace-prior heuristic 与 oracle。

3. **上弱学习器前先扩数据**  
   MLP/XGBoost 可以作为 Day8-Day10 的方向，但现在 3 个 block、18 条 candidate summary 太少。应先扩到至少几十个 block/shape，再做弱学习器。直接上图模型不建议。

## 是否建议进入 Day6-Day7，以及进入时应修改什么

建议进入 Day6-Day7，但要带条件：

- Day6 可以做 simple heuristic，但不能把当前 3 个 block 的结果当成最终结论。
- Day6 的 heuristic 必须按 shape/seq 分组报告，不能只给总平均。
- Day7 的可行性判断必须把 b2 标为弱样本，不能只看 b0/b1。
- 在 Day6 同时准备扩 candidate space 的计划；如果 heuristic 接近 oracle，优先扩 candidate，不要上模型。
- Day7 结论中必须明确：当前证据支持“问题存在的初步信号”，不支持“图模型必要”。

## 使用的关键文件

- `day1/day1_scope.md`
- `day2/data_schema.md`
- `day2/fx_nodes.csv`
- `day3/rewrite_rules.md`
- `day3/candidate_plans.json`
- `day4/block_run_summary.json`
- `day5/compile_reset/profile_run_summary.json`
- `day5/compile_reset/candidate_summary.csv`
- `day5/compile_reset/block_backend_summary.csv`
- `day5/compile_reset/raw_profile.csv`
- `day5/LOG.md`
