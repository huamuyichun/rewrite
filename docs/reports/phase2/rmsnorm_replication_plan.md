# RMSNorm Adaptive Replication Plan

## 目标

本计划只复测现有 8 个 RMSNorm discovery groups 中信息量最高的 6 个，用于判断
该 family 应保留为主要 context-sensitive family、降为 control/diagnostic，还是停止
扩大。当前离线证据更支持“降级”：没有唯一 strict semantic winner，五个核心
semantic plans 在 8/8 groups 都是 possible winner；global best fixed 的 raw P90/max
regret 仅 0.461%/0.580%，production 到 noise-aware oracle 的 gain 为 0。

复测不训练任何模型，不修改 2% noise floor 或 0.5% monitor gate，不合并其他服务器
的绝对 latency。所有新 session 必须属于
`bear-a100-b0f2d831-cu129-torch210-driver560-20260720` domain。

## 选择结果

| original | group | 选择理由 | target sessions |
| --- | --- | --- | ---: |
| `rms_d01` | norm-only decode bs1/t1 | default-to-point gain 2.375%；paired CI 1.574%-2.912% 跨越 2% gate | 3 |
| `rms_d02` | norm-only decode bs8/t1 | 覆盖较大 decode；max class CV 13.217%，需检查 session drift | 3 |
| `rms_d03` | residual-SiLU decode bs1/t1 | gain 2.211%；CI -0.295%-2.621%；同类 timing spread 2.300% | 3 |
| `rms_d04` | residual-SiLU decode bs8/t1 | 最大 best-worst spread 7.538%；point-best 从 square-mul 变为 square-pow | 3 |
| `rms_p02` | norm-only prefill bs1/s1024 | 最大 norm-only prefill；baseline tie；同类 timing spread 4.878% | 3 |
| `rms_p03` | residual-SiLU prefill bs1/s128 | 覆盖 prefill/context；两条同类 timing warning，最高 3.896% | 3 |

不复测：

- `rms_p01`：s128 norm-only 的 point gain 1.266%，由同 shape、诊断信号更强的
  `rms_p03` 覆盖；保留原 session 作为单 session tie/ambiguous evidence。
- `rms_p04`：当前无 same-class warning，baseline-vs-point CI 完全位于 2% floor 内；
  最大 prefill 由 `rms_p02` 覆盖，square-pow point-best 由 `rms_d04` 覆盖。

当前不存在 provisional strict semantic winner。“square-mul/square-pow point-best”只
是 P50 排序，不能写成 strict winner；复测不以制造唯一 winner 为目标。

## 执行协议

每个选中 group 新增两个完整独立 session，使其与原 session 合计为三个：

| group short name | second session | third session | protocol |
| --- | --- | --- | --- |
| `d01` | `rms_d01_r02` | `rms_d01_r03` | `phase2_discovery_decode_v1.json` |
| `d02` | `rms_d02_r02` | `rms_d02_r03` | `phase2_discovery_decode_v1.json` |
| `d03` | `rms_d03_r02` | `rms_d03_r03` | `phase2_discovery_decode_v1.json` |
| `d04` | `rms_d04_r02` | `rms_d04_r03` | `phase2_discovery_decode_v1.json` |
| `p02` | `rms_p02_r02` | `rms_p02_r03` | `phase2_discovery_prefill_v1.json` |
| `p03` | `rms_p03_r02` | `rms_p03_r03` | `phase2_discovery_prefill_v1.json` |

每次 invocation 必须满足：

1. 独立 Python process 和独立 `cold_session` Inductor cache。
2. `CUDA_DEVICE_ORDER=PCI_BUS_ID`、`CUDA_VISIBLE_DEVICES=0`，PyTorch UUID 与
   NVML UUID 都是 `GPU-b0f2d831-6a1a-c820-f388-431148eabf25`。
3. timing window 内 monitor-off，NVML 只做 boundary snapshots。
4. 启动前同时检查 GPU utilization 和 compute PID；有 foreign PID 时不运行、不排队。
5. equivalence/alias 全通过，8 FX / 8 lowered / 6 execution；任何变化先停止扩展。
6. source clean；registry 只允许已有 registry 自身变更，不覆盖任何 session 目录。

统一命令模板：

```bash
source .env.migration
conda activate "$LOCAL_DATA_ROOT/envs/rewrite"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export RUN_ID="phase2_rmsnorm_replication_$(date +%Y%m%d_%H%M%S)"

python scripts/run_phase1_audit.py \
  --rewrites configs/rewrites/rmsnorm_bounded_v1.json \
  --workloads configs/workloads/phase2_rmsnorm_discovery_v1.json \
  --protocol configs/profiling/<decode-or-prefill>.json \
  --group-id <full-group-id> \
  --run-id "$RUN_ID" \
  --session-id <planned-session-id> \
  --output-root "$REWRITE_ARTIFACT_ROOT/phase2" \
  --registry "$REWRITE_REGISTRY_PATH"
```

## 两阶段门禁

### R1：第二个 session

先依次运行 6 个 `_r02` session，每组完成后审计 status、equivalence、retention、
fingerprint schema、mapping、boundary clock、foreign PID 和 contamination。全部完成后
立即重跑 offline aggregation。

任一组出现以下情况时，暂停该组 R2，并先调查：

- execution fingerprint 或 semantic-plan-to-class mapping 改变；
- generated code、stride/layout、cache policy 或环境域不一致；
- foreign PID、contaminated round 或 boundary clock 明显异常；
- eager/compiled/alias validation 失败。

### R2：第三个 session

只有 R1 clean 且 fingerprint/mapping 稳定的组才运行 `_r03`。完成后用初始 run 与
replication run 共同聚合，跨 session 统计以 session 为独立单位，不把其他服务器 raw
samples 合入。

## 跨 Session 判定

最终逐组报告：

- fingerprint 与 execution-class mapping stability；
- noise-aware best-set reproducibility；
- pairwise strict/tie/ambiguous 及 order reproducibility；
- baseline-to-best gain 的 median/range；
- class/session P50 drift；
- contaminated session ratio；
- same-class timing warning 是否复现。

family 决策预注册如下：

- **A 保留**：至少两个 semantic plans 在不同 context 中产生可复现 strict best，且
  global fixed 有非平凡 tail regret，简单 phase/context/shape 规则不能接近 oracle。
- **B 降级**：fingerprint/mapping 稳定且存在可区分 execution classes，但一个 fixed
  plan 或简单规则保持接近 oracle，global fixed P90 regret 明显低于 3%，production
  的 noise-aware gap 很低。
- **C 停止扩大**：核心 classes 大多无法超过 noise，production 已接近 oracle，或
  equivalence/fingerprint/mapping 无法稳定。

在 RMSNorm decision 形成前，不启动 9 个 MLP control groups。
