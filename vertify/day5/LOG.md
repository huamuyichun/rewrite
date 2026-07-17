# Day 5 日志

日期：2026-05-26

## 1. 做了什么

创建 Day 5 工作目录：

```text
/pub/data/hjwz/rewrite/vertify/day5
```

创建 profiling 脚本：

```text
profile_candidates.py
```

该脚本复用 Day 4 的 candidate module 构造逻辑，对 Day 4 的 3 个 block、每个 block 6 个 candidates 做 latency 测量。

## 2. 目的是什么

Day 5 的目的不是判断哪个 selector 好，也不是训练模型，而是接入 profiling 流水线。

需要确认：

```text
candidate latency 能否被稳定测出来，并保存 raw profile 与 summary。
```

## 3. 当前 profiling 协议

默认协议：

```text
backend_mode = compile
warmup = 10
measure = 30
timer = torch.cuda.Event + synchronize
```

说明：

- `torch.compile` 的第一次 compile/prime 调用不计入 warmup/measure；
- compile prime time 单独记录；
- raw profile 中同时保留 warmup 和正式测量；
- summary 只用正式测量计算。

## 4. 缓存位置

Day 5 显式使用本地盘 Inductor cache：

```text
TORCHINDUCTOR_CACHE_DIR=/pub/data/hjwz/.cache/torchinductor
```

目的：

```text
避免 torch.compile 高频缓存写回 /home/hjwz。
```

## 5. 当前边界

Day 5 不做：

- heuristic selector；
- oracle selector 主表；
- learning model；
- chitu 接入；
- 论文级完整数据集。

这些属于 Day 6 以后。

## 6. 实际运行与修正

第一次运行 compile profiling 时，脚本产出了完整数据，但 PyTorch Dynamo 输出了 recompile limit warning。

原因判断：

```text
Day 4 的 candidate module 使用少量共享 Python forward code object，
不同 candidate 通过 activation / multiply / split_mode 属性走不同分支，
torch.compile 对这些属性生成 guard，连续编译多个 candidate 时触发 recompile limit warning。
```

处理方式：

```text
在每个 candidate 调用 torch.compile 前执行 torch._dynamo.reset()。
```

修复后重新运行 compile profiling，输出到：

```text
/pub/data/hjwz/rewrite/vertify/day5/compile_reset
```

另外补跑 eager profiling，输出到：

```text
/pub/data/hjwz/rewrite/vertify/day5/eager
```

主口径采用 `compile_reset`，因为它对应 Day 1 固定的 `torch.compile(inductor)` backend，且没有 recompile limit warning。

## 7. Compile Profiling 结果

运行命令：

```text
conda activate rewrite_miniexp
TORCHINDUCTOR_CACHE_DIR=/pub/data/hjwz/.cache/torchinductor \
python /pub/data/hjwz/rewrite/vertify/day5/profile_candidates.py \
  --out-dir /pub/data/hjwz/rewrite/vertify/day5/compile_reset \
  --backend-modes compile \
  --warmup 10 \
  --measure 30
```

汇总：

```text
status = ok
backend_mode = compile
num_blocks = 3
num_candidates_per_block = 6
num_block_backend_cases = 3
num_raw_rows = 720
num_summary_rows = 18
torch = 2.10.0+cu129
device = NVIDIA A100 80GB PCIe
```

每个 block 的结果：

```text
b0_seq128_h1024_i4096:
  baseline_p50 = 0.096768 ms
  best = p5_fused_chunk_manual_silu
  best_p50 = 0.086016 ms
  spread = 0.125000
  median_cv = 0.046155
  p90_cv = 0.053162
  winner_flip = false

b1_seq128_h768_i3072:
  baseline_p50 = 0.097280 ms
  best = p3_fused_chunk_silu
  best_p50 = 0.089600 ms
  spread = 0.125714
  median_cv = 0.047315
  p90_cv = 0.058797
  winner_flip = false

b2_seq512_h768_i3072:
  baseline_p50 = 0.108544 ms
  best = p3_fused_chunk_silu
  best_p50 = 0.104448 ms
  spread = 0.039216
  median_cv = 0.028807
  p90_cv = 0.031112
  winner_flip = true
```

## 8. Eager Profiling 补充结果

输出目录：

```text
/pub/data/hjwz/rewrite/vertify/day5/eager
```

汇总：

```text
status = ok
backend_mode = eager
num_blocks = 3
num_candidates_per_block = 6
num_raw_rows = 720
num_summary_rows = 18
```

Eager 结果只作为补充，不作为 Day 5 主证据。

## 9. Day 5 结论

Day 5 验收通过。

已经完成：

1. profiling 脚本：`profile_candidates.py`；
2. raw profile：`compile_reset/raw_profile.csv` 和 `eager/raw_profile.csv`；
3. candidate latency 表：`compile_reset/candidate_summary.csv` 和 `eager/candidate_summary.csv`；
4. block/backend 稳定性表：`compile_reset/block_backend_summary.csv` 和 `eager/block_backend_summary.csv`；
5. warmup / measure / CUDA synchronize / p50 / mean / std / CV / winner flip 均已记录；
6. Inductor cache 已放在 `/pub/data/hjwz/.cache/torchinductor`。

当前初步观察：

- compile backend 下 3 个 block 都存在 candidate latency spread；
- b0、b1 的 first-half / second-half winner 稳定；
- b2 的 winner 出现 flip，说明该 shape 下候选差异更弱或测量噪声相对更大；
- 这些结果只说明 profiling 流水线能跑通，不等同于 Day 6/Day 7 的 selector 结论。

下一步 Day 6 应做：

```text
基于 Day 5 的 profiling 结果，提前插入 simple heuristic baseline，
比较 heuristic 选择与 baseline/default、profile-best candidate 的差距。
```
