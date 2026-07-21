# RMSNorm Family 正式决策

## 决策

**B. 降为 control/diagnostic family。**

RMSNorm 保留为 equivalence、lowering-collapse、fingerprint 和 measurement diagnostic family；不作为 learned context-sensitive selector 的主要训练空间，不再通过扩相似 shape 或 rewrite 数量规避该结论。

## 证据

- 8/8 groups 均为 8 FX → 6 execution，retention 75.000%；20 个正式 session 全部通过 provenance，新增 12 个 session 无 aborted/contaminated。
- aggregate pair 为 57 strict / 14 tie / 49 ambiguous；每组有 5-6 个 class 参与 strict pair，execution diversity 和性能差异真实存在。
- 唯一 strict semantic winner 数为 0；五个核心 semantic plans 在 8/8 groups 都是 possible winner。point winner 的变化没有形成跨 context 可复现的 strict semantic winner exchange。
- global fixed 为 rmsnorm.decompose_square_pow (`sem_1f8ff2cd110a45ba`)，raw P50/P90/max regret 0.000%/0.523%/0.787%，noise-aware max 0.000%。
- decode fixed 为 rmsnorm.decompose_square_mul+rmsnorm.reassociate_scale (`sem_584eba078e62c9da`)，raw max regret 0.103%；prefill fixed 为 rmsnorm.decompose_square_pow (`sem_1f8ff2cd110a45ba`)，raw max regret 0.000%。
- 最简单近似 oracle 规则 `decode_vs_prefill` 的 raw P90/max regret 为 0.031%/0.103%。
- production/default 到 point oracle 的跨 group median/P90/max gain 为 1.478%/2.732%/3.947%；到 noise-aware oracle 为 0.000%/0.000%/0.000%。production 在每组 noise-aware best set 中。
- same-class 短 diagnostic warning 共 12 条；fingerprint 和 semantic→execution mapping 跨 session 全部稳定，因此它们作为 measurement variability 记录，不解释为 context-sensitive codegen。

## 选择性复测

| group | sessions | point winner | best-set exact | pair order | default→point median [min,max] | max class drift | mapping/fingerprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `phase2_rmsnorm_norm_only_decode_bs1_t1_bf16` | 3 | 100.000% | 100.000% | 93.333% | 2.375% [1.941%, 2.922%] | 3.505% | yes |
| `phase2_rmsnorm_norm_only_decode_bs8_t1_bf16` | 3 | 66.667% | 66.667% | 93.333% | 1.328% [1.231%, 1.587%] | 5.132% | yes |
| `phase2_rmsnorm_norm_only_prefill_bs1_s1024_bf16` | 3 | 66.667% | 66.667% | 60.000% | 0.667% [0.000%, 3.947%] | 11.538% | yes |
| `phase2_rmsnorm_residual_silu_decode_bs1_t1_bf16` | 3 | 100.000% | 100.000% | 93.333% | 1.797% [1.759%, 2.211%] | 5.299% | yes |
| `phase2_rmsnorm_residual_silu_decode_bs8_t1_bf16` | 3 | 66.667% | 100.000% | 93.333% | 1.793% [1.641%, 2.255%] | 3.577% | yes |
| `phase2_rmsnorm_residual_silu_prefill_bs1_s128_bf16` | 3 | 100.000% | 66.667% | 86.667% | 1.316% [1.299%, 1.333%] | 3.165% | yes |

`rms_p02` 的 point gain 为 0%/3.947%/0.667%，pair-order reproducibility 只有 60%，第三次没有复现第二次的大 gap；其 fingerprint、mapping、clock 和污染门禁均正常。这是 session variability 证据，不是稳定 winner exchange。

## MLP 后续

RMSNorm decision 已完成，允许开始 9 个 MLP control groups。每组先运行一个独立 screening session，先做 equivalence 和 FX/lowered/execution dedup，只正式 profile execution-unique candidates。screening 聚合后才选择信息量高的 groups 做 adaptive replication；不机械重复 19 个 FX candidates，不训练 selector。

Phase 2 exit gate 仍为 `pending_mlp_control_discovery`。
