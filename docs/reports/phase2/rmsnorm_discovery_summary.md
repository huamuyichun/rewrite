# RMSNorm Discovery 离线聚合

## 结论

- 分析域：`bear-a100-b0f2d831-cu129-torch210-driver560-20260720`；迁移 tag：`migration-a100-20260720`。
- 8 FX → 6 execution 的 retention 在全部 8 个 group 中稳定。
- best-worst spread 超过锁定 2% floor 的 group：8/8；存在至少一条 strict pair 的 group：8/8。
- 每组参与 strict pair 的 execution class 数：[6, 5, 6, 5, 6, 6, 5, 4]；noise-aware best set 平均包含 3.375 个 execution class。
- noise-aware best semantic-plan set 随 group 变化，但所有 group 共同保留 5 个 possible plans，唯一 strict semantic winner 数为 0。
- global best fixed：rmsnorm.decompose_square_pow (`sem_1f8ff2cd110a45ba`)；possible-win share 100.000%，raw P50/P90/max regret 为 0.000%/0.461%/0.580%，noise-aware max 为 0.000%。
- production/default 到 point oracle 的 median/P90/max gain 为 1.299%/2.260%/2.375%；到 noise-aware oracle 为 0.000%/0.000%/0.000%。
- 初步 context-sensitive selection 证据：不足；fixed/simple policy 已接近 oracle，复测用于确认降级结论。

所有绝对 latency 只在该 hardware/environment domain 内聚合；旧服务器仅可比较 normalized gain、排序与 fingerprint。锁定的 2% noise floor 未修改。

## 统计口径

`semantic_plan_id` 由 family、canonical rewrite trace 与语义参数的 canonical JSON 哈希生成。它跨 workload 稳定，再按 group 多对一映射到 execution class。pairwise 相对差异按 blocked round 配对 bootstrap；95% CI 整体越过 ±2% 才是 strict，完全落在区间内是 tie，其余是 ambiguous。noise-aware regret 对 best set 内计划记 0。

## Group 结果

| group | sessions | enum/valid/FX/lowered/exec | strict/tie/ambiguous | best-set classes | spread | default→point | default→noise-aware | fingerprint stable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `phase2_rmsnorm_norm_only_decode_bs1_t1_bf16` | rms_d01 | 8/8/8/8/6 | 9/4/2 | 3 | 6.702% | 2.375% | 0.000% | yes |
| `phase2_rmsnorm_norm_only_decode_bs8_t1_bf16` | rms_d02 | 8/8/8/8/6 | 4/1/10 | 4 | 5.695% | 1.231% | 0.000% | yes |
| `phase2_rmsnorm_norm_only_prefill_bs1_s1024_bf16` | rms_p02 | 8/8/8/8/6 | 8/1/6 | 3 | 6.098% | 0.000% | 0.000% | yes |
| `phase2_rmsnorm_norm_only_prefill_bs1_s128_bf16` | rms_p01 | 8/8/8/8/6 | 5/0/10 | 3 | 6.962% | 1.266% | 0.000% | yes |
| `phase2_rmsnorm_residual_silu_decode_bs1_t1_bf16` | rms_d03 | 8/8/8/8/6 | 8/1/6 | 3 | 6.030% | 2.211% | 0.000% | yes |
| `phase2_rmsnorm_residual_silu_decode_bs8_t1_bf16` | rms_d04 | 8/8/8/8/6 | 9/2/4 | 3 | 7.538% | 1.641% | 0.000% | yes |
| `phase2_rmsnorm_residual_silu_prefill_bs1_s1024_bf16` | rms_p04 | 8/8/8/8/6 | 6/1/8 | 4 | 6.494% | 1.299% | 0.000% | yes |
| `phase2_rmsnorm_residual_silu_prefill_bs1_s128_bf16` | rms_p03 | 8/8/8/8/6 | 4/0/11 | 4 | 5.844% | 1.299% | 0.000% | yes |

每个 execution class 的 candidates、semantic plans、raw sample count、P50/mean/CV、bootstrap CI 和所有 relative-difference CI 保存在对应 JSON；CSV 为一行一个 group-local execution class。

## Semantic Plans

| semantic plan | label | strict win | possible win | fractional win | raw P50 | raw P90 | raw max | noise max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sem_1f8ff2cd110a45ba` | rmsnorm.decompose_square_pow | 0.000% | 100.000% | 18.750% | 0.000% | 0.461% | 0.580% | 0.000% |
| `sem_286a68462afba0aa` | rmsnorm.decompose_square_pow+rmsnorm.reassociate_scale | 0.000% | 100.000% | 18.750% | 0.000% | 0.461% | 0.580% | 0.000% |
| `sem_584eba078e62c9da` | rmsnorm.decompose_square_mul+rmsnorm.reassociate_scale | 0.000% | 100.000% | 18.750% | 0.000% | 0.677% | 1.299% | 0.000% |
| `sem_f9305041e4a24aea` | rmsnorm.decompose_square_mul | 0.000% | 100.000% | 18.750% | 0.000% | 0.677% | 1.299% | 0.000% |
| `sem_e8052199546100d5` | production_default | 0.000% | 100.000% | 18.750% | 1.299% | 2.260% | 2.375% | 0.000% |
| `sem_6217483aedc0e5e0` | rmsnorm.decompose_square_pow+rmsnorm.flatten_hidden_rows | 0.000% | 37.500% | 6.250% | 4.313% | 5.706% | 5.858% | 5.858% |
| `sem_9d4de2b6c6ee960c` | rmsnorm.decompose_square_mul+rmsnorm.flatten_hidden_rows | 0.000% | 0.000% | 0.000% | 5.553% | 6.935% | 6.962% | 6.962% |
| `sem_5bb3ef804d6e34e0` | rmsnorm.flatten_hidden_rows | 0.000% | 0.000% | 0.000% | 6.213% | 6.953% | 7.538% | 7.538% |

## Fixed Baselines

| scope | best fixed | raw P50 | raw P90 | raw max | noise max | possible win |
| --- | --- | --- | --- | --- | --- | --- |
| all | rmsnorm.decompose_square_pow (`sem_1f8ff2cd110a45ba`) | 0.000% | 0.461% | 0.580% | 0.000% | 100.000% |
| decode | rmsnorm.decompose_square_mul+rmsnorm.reassociate_scale (`sem_584eba078e62c9da`) | 0.000% | 0.287% | 0.410% | 0.000% | 100.000% |
| prefill | rmsnorm.decompose_square_pow (`sem_1f8ff2cd110a45ba`) | 0.000% | 0.000% | 0.000% | 0.000% | 100.000% |

fractional winner entropy：1.742640（normalized 0.972586）。tie group 没有被强制指定唯一 winner。

## Top-k Oracle

| k | semantic plan portfolio | raw P50 | raw P90 | raw max | noise max |
| --- | --- | --- | --- | --- | --- |
| 1 | rmsnorm.decompose_square_pow (`sem_1f8ff2cd110a45ba`) | 0.000% | 0.461% | 0.580% | 0.000% |
| 2 | rmsnorm.decompose_square_pow (`sem_1f8ff2cd110a45ba`), rmsnorm.decompose_square_mul+rmsnorm.reassociate_scale (`sem_584eba078e62c9da`) | 0.000% | 0.000% | 0.000% | 0.000% |
| 3 | rmsnorm.decompose_square_pow (`sem_1f8ff2cd110a45ba`), rmsnorm.decompose_square_pow+rmsnorm.reassociate_scale (`sem_286a68462afba0aa`), rmsnorm.decompose_square_mul+rmsnorm.reassociate_scale (`sem_584eba078e62c9da`) | 0.000% | 0.000% | 0.000% | 0.000% |

## 简单规则诊断

| rule | buckets | raw P50 | raw P90 | raw max | noise max |
| --- | --- | --- | --- | --- | --- |
| global_fixed | 1 | 0.000% | 0.461% | 0.580% | 0.000% |
| decode_vs_prefill | 2 | 0.000% | 0.123% | 0.410% | 0.000% |
| context_type | 2 | 0.000% | 0.015% | 0.050% | 0.000% |
| exact_shape_bucket | 4 | 0.000% | 0.123% | 0.410% | 0.000% |
| batch_threshold<=1 | 2 | 0.000% | 0.461% | 0.580% | 0.000% |
| sequence_length_threshold<=1 | 2 | 0.000% | 0.123% | 0.410% | 0.000% |
| token_threshold<=128 | 2 | 0.000% | 0.123% | 0.410% | 0.000% |

这些规则只在当前 8 个 group 上做同集 diagnostic，不是训练结果，也不代表 held-out 泛化。exact shape bucket 仍让两个 context 共用一个固定计划，避免一组一个规则的无意义拟合。

## Provenance

| session | commit | config | GPU UUID | Triton | binding | audit |
| --- | --- | --- | --- | --- | --- | --- |
| `rms_d01` | `72eb9705ce6a` | `c5ebc5a5c455` | GPU-b0f2d831-6a1a-c820-f388-431148eabf25 | 3.6.0 | frozen_domain_supplement | pass |
| `rms_d02` | `72eb9705ce6a` | `0bb8adbbe83d` | GPU-b0f2d831-6a1a-c820-f388-431148eabf25 | 3.6.0 | frozen_domain_supplement | pass |
| `rms_d03` | `72eb9705ce6a` | `8eb16d441286` | GPU-b0f2d831-6a1a-c820-f388-431148eabf25 | 3.6.0 | frozen_domain_supplement | pass |
| `rms_d04` | `72eb9705ce6a` | `3fb8196e9ee5` | GPU-b0f2d831-6a1a-c820-f388-431148eabf25 | 3.6.0 | frozen_domain_supplement | pass |
| `rms_p01` | `72eb9705ce6a` | `493f89186a8a` | GPU-b0f2d831-6a1a-c820-f388-431148eabf25 | 3.6.0 | frozen_domain_supplement | pass |
| `rms_p02` | `72eb9705ce6a` | `03995a020599` | GPU-b0f2d831-6a1a-c820-f388-431148eabf25 | 3.6.0 | frozen_domain_supplement | pass |
| `rms_p03` | `72eb9705ce6a` | `57eebbc28f96` | GPU-b0f2d831-6a1a-c820-f388-431148eabf25 | 3.6.0 | frozen_domain_supplement | pass |
| `rms_p04` | `72eb9705ce6a` | `25a16a528407` | GPU-b0f2d831-6a1a-c820-f388-431148eabf25 | 3.6.0 | frozen_domain_supplement | pass |

历史 discovery manifest 未直接写入 Triton 和 CUDA device order；二者由冻结的 environment-domain record 与迁移报告补充绑定。新 runner 已直接记录这些字段。

## 限制

当前每组只有一个完整独立 session。CI 反映 blocked-round measurement uncertainty，尚不能证明跨 session 可复现性；family 保留/降级/淘汰决定必须等待 adaptive replication 完成。
