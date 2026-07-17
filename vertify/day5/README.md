# Day 5: Profiling Pipeline

日期：2026-05-26

## 目的

Day 5 的目标是接入真实 latency profiling 流水线。

本阶段回答：

```text
Day 4 生成的 candidates 是否能被稳定测量，并产出 raw profile 与 per-candidate latency summary？
```

本阶段不做：

- heuristic selector；
- oracle selector 主表；
- 学习模型；
- chitu 接入；
- 论文级完整数据集。

## 输入

- Day 3 candidate plans：`../day3/candidate_plans.json`
- Day 4 block specs：`../day4/block_specs.json`
- Day 4 module 实例化逻辑：`../day4/instantiate_candidates.py`

## Profiling 协议

默认：

```text
backend_mode = compile
warmup = 10
measure = 30
dtype = fp16
batch_size = 1
```

计时方式：

- CUDA 环境下使用 `torch.cuda.Event`；
- 每次计时前后执行 CUDA synchronize；
- `torch.compile` 的第一次 compile/prime 调用不计入 warmup/measure；
- compile prime time 单独记录为 `compile_prime_time_ms`。

缓存策略：

```text
TORCHINDUCTOR_CACHE_DIR=/pub/data/hjwz/.cache/torchinductor
```

避免把 Inductor 编译缓存写回 `/home/hjwz`。

## 输出

运行 `profile_candidates.py` 后生成：

- `raw_profile.csv`：每次 warmup/measure 的原始 latency；
- `candidate_summary.csv`：每个 block/backend/candidate 的 p50/mean/std/CV/rank；
- `block_backend_summary.csv`：每个 block/backend 的 spread、CV、winner flip；
- `profile_run_summary.json`：本次 profiling 总结；
- `block_backend_outputs/`：每个 block/backend 的局部 raw/summary。

## 验收标准

Day 5 通过条件：

1. 每个 candidate 都能完成 warmup 和正式测量；
2. 产出 raw profile；
3. 产出 candidate latency summary；
4. 记录 p50、mean、std、CV；
5. 记录 first-half vs second-half winner flip；
6. profiling 日志中说明目的、动作和结论。
