# 给下一轮 Codex 的接力 Prompt

你正在接手 `/home/hejwz/rewrite`。这是当前服务器上的实际仓库路径；
`/pub/data/hjwz/rewrite` 在本机不存在。先完整阅读：

1. `/home/hejwz/AGENTS.md`
2. `docs/SERVER_MIGRATION_GUIDE.md`
3. `docs/rewrite_research_plan.md`
4. 本文件
5. `README.md`、`environment/README.md`、`docs/artifact_policy.md`

本轮工作的核心问题是：

> RMSNorm 和 MLP 的多个语义等价 rewrite，在 Inductor 后是否形成性能可区分、
> 会随真实 workload/context 稳定交换优劣的执行方案，并且这种变化是否无法由
> 单一 fixed plan 或简单规则解决？

不要恢复旧 GNN、`vertify`、`rewrite_miniexp` 或历史 pilot 路线。在 Phase 2
exit decision 前禁止训练 GNN、GraphSAGE、DeepSets、tree ranker、MLP selector、
ranking model 或任何其他 selector。

## 1. 当前暂停点

会话于 2026-07-21 UTC 暂停在以下位置：

- 迁移和 Phase 1 recalibration 已完成。
- 8 个 RMSNorm discovery groups 的离线聚合、选择性复测和 family decision 已完成。
- RMSNorm 正式结论为 **B：降为 control/diagnostic family**。
- MLP 的首次 canary 暴露 fingerprint/cache bug，已保留但排除。
- bug 已修复并推送，修复后的 `mlp_d01_s01` canary 已完整通过。
- 其余 8 个 MLP screening groups 尚未运行。
- MLP 离线聚合、adaptive replication、三份 Phase 2 总报告和 README 最终环境更新
  尚未完成。
- 当前没有 GPU 实验在运行。

先执行：

```bash
cd /home/hejwz/rewrite
cat /home/hejwz/AGENTS.md
cat docs/SERVER_MIGRATION_GUIDE.md
git status --short --branch
git log --oneline --decorate -10
git rev-parse HEAD
git rev-parse origin/main
```

不要重复 RMSNorm 实验，也不要重新运行已经有效的 MLP canary，除非后续 adaptive
replication 明确选择它。

## 2. 服务器和环境冻结状态

迁移稳定 tag：`migration-a100-20260720`

迁移基线 commit：

```text
fa0e4471b9cb44b5d83ebf321585597c11ef579f
```

runner fingerprint 修复 commit：

```text
76c6cd1287df59dbe0a86888c5271766fc6bd47e
```

正式 hardware/environment domain：

```text
bear-a100-b0f2d831-cu129-torch210-driver560-20260720
```

冻结配置：

- Python 3.12.13
- PyTorch 2.10.0+cu129
- CUDA runtime 12.9
- Triton 3.6.0
- NVIDIA driver 560.35.05
- GPU 0：NVIDIA A100 80GB PCIe
- GPU UUID：`GPU-b0f2d831-6a1a-c820-f388-431148eabf25`
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`
- cache policy：`cold_session`
- candidate compile policy：`force_disable_caches_per_candidate`
- timing window：`monitor_mode=off`
- timing 边界：NVML snapshots only
- noise floor：2%，禁止修改
- monitor self-effect gate：0.5%，禁止修改

环境入口：

```bash
cd /home/hejwz/rewrite
source .env.migration
conda activate "$LOCAL_DATA_ROOT/envs/rewrite"
```

实际 Conda prefix 为：

```text
/home/hejwz/rewrite/.local-data/envs/rewrite
```

本机 `/home` 所在文件系统在暂停时约 93% 已用、剩余约 34 GiB。继续生成多个
Inductor cache 前先复查空间，但不要删除已有 raw evidence。所有写入保持在
`/home/hejwz` 或 `/tmp`，不要使用 sudo、修改系统目录或影响其他用户。

Qwen 权重和旧服务器 raw Phase 1 artifacts 没有同步。这两项不是当前 microbenchmark
主线的阻塞项，不要为了消除 warning 下载权重或中断 MLP discovery。

不同服务器的 absolute latency raw samples 禁止合并。只可跨域比较 normalized gain、
排序和 fingerprint；绝对 latency 必须按 hardware/environment domain 分开报告。

## 3. Git 和验证状态

暂停前最后一个已推送代码 commit 是 `76c6cd1`，当时 `HEAD == origin/main`。
修复后的 MLP canary 随后向 `artifacts/registry.jsonl` 追加了一条有效 session 记录，
所以接手时应先检查该 registry 记录和本接力文档是否已经提交。

最近主要 commits：

```text
76c6cd1 fix: 阻止缺失fingerprint进入正式profiling
7bd10fb feat: 完成RMSNorm复测与family决策
9fb4b76 feat: 完成RMSNorm第二会话复测审计
c71171d docs: 制定RMSNorm选择性复测计划
5a67f05 feat: 完成RMSNorm离线聚合分析
fa0e447 feat: 完成迁移后的RMSNorm发现阶段
72eb970 feat: 完成目标服务器迁移与重校准
```

`76c6cd1` 的 CPU 验证结果：

- pytest：38 passed
- compileall：passed
- `git diff --check`：passed
- 唯一 warning 是环境中没有可选 NumPy，PyTorch 导入时打印 warning；不要为它偏离主线

## 4. 不可改变的实验边界

- 不训练任何 selector 或 ranking model。
- 不修改 2% noise floor。
- 不修改 0.5% monitor gate。
- 正式 timing 内保持 monitor-off。
- 每个正式 session 使用独立 Python process 和独立 cold cache。
- 每次只使用一张无 foreign PID 的明确空闲 GPU。
- 不同时启动多个 GPU 任务。
- 不把 execution hash 当作跨 group 的 fixed plan ID。
- 跨 group fixed policy 必须使用 `semantic_plan_id`。
- `candidate_id` 只作 group-local provenance。
- `execution_class_id` 只作具体 group 内去重、oracle 和 latency 比较。
- tie group 不得强制指定唯一 winner。
- 不合并不同服务器的绝对 latency samples。
- 不下载 Qwen 权重，不进入完整 Chitu/model serving 验证。
- 不通过增加无意义 rewrite 或修改阈值规避负面结论。

GPU 工具在 Codex filesystem sandbox 内曾错误报告 `cuda_available=false`，而同一时刻
沙箱外检查通过。GPU gate 和正式实验需要使用获准的沙箱外执行；这不是修改 driver
或系统配置的理由。每次仍需重新检查：

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu,clocks.sm \
  --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits
```

随后只暴露 GPU 0，并运行：

```bash
source .env.migration
export CUDA_VISIBLE_DEVICES=0
.local-data/envs/rewrite/bin/python scripts/check_migration.py \
  --require-gpu --require-clean-git
```

Qwen/旧 raw warning 可忽略；GPU、Git、UUID、foreign PID 或 source state gate 失败则停止。

## 5. RMSNorm 已完成结果

正式 discovery groups：

```text
rms_d01 rms_d02 rms_d03 rms_d04
rms_p01 rms_p02 rms_p03 rms_p04
```

初始 discovery run：

```text
phase2_rmsnorm_discovery_20260720_030834
```

选择性 replication run：

```text
phase2_rmsnorm_replication_20260721_031121
```

新增 12 个 clean replication sessions：

```text
rms_d01_r02 rms_d01_r03
rms_d02_r02 rms_d02_r03
rms_d03_r02 rms_d03_r03
rms_d04_r02 rms_d04_r03
rms_p02_r02 rms_p02_r03
rms_p03_r02 rms_p03_r03
```

没有 aborted 或 contaminated RMSNorm session。所有正式 session 均为 monitor-off、
无 foreign PID、映射和 fingerprint 稳定。

最终聚合：

- 8/8 groups 均为 8 valid、8 FX unique、6 execution unique。
- execution retention 全部为 75%。
- 20 个正式 sessions。
- pair 汇总：57 strict、14 tie、49 ambiguous。
- 8/8 groups 有至少一个 strict pair，但没有唯一 strict semantic winner。
- noise-aware best set 平均大小：3.125。
- 五个核心 semantic plans 在全部 groups 中都仍是 possible winner。
- global fixed：`sem_1f8ff2cd110a45ba`，label=`square-pow`。
- global fixed raw regret P50/P90/max：0% / 0.523% / 0.787%。
- global fixed noise-aware regret：0%。
- decode/prefill simple split raw max regret：0.103%。
- production-to-point-oracle median/P90/max gain：1.478% / 2.732% / 3.947%。
- production-to-noise-aware-oracle gain：0%。
- `rms_p02` 的 session gain 0% / 3.947% / 0.667%，未形成可复现 winner exchange。

正式决定：

```text
B：降为 control/diagnostic family
```

理由是 execution diversity 和 lowering-collapse 有诊断价值，但一个 fixed semantic plan
或极简单 decode/prefill rule 已处于 noise-aware oracle 内，learning-based selector 没有
必要性。不要扩大相似 RMSNorm shapes，也不要重复修改结论。

关键报告：

- `docs/reports/phase2/rmsnorm_discovery_summary.md/json/csv`
- `docs/reports/phase2/rmsnorm_discovery_summary_r1.md/json/csv`
- `docs/reports/phase2/rmsnorm_replication_plan.md`
- `docs/reports/phase2/rmsnorm_family_decision.md/json`

## 6. MLP 首次无效 canary

首次 run：

```text
run_id: phase2_mlp_screening_20260721_034818
session_id: mlp_d01
artifact: .local-data/artifacts/phase2/phase2_mlp_screening_20260721_034818/mlp_d01
source: 7bd10fb2140022242471aaf7bfc7912cb37630aa
```

它表面报告 19 valid、19 FX、6 lowered、6 execution 和约 7.04% gain，但实际 formal
profile 有 10 个 units。四个 candidates 没有 trace/fingerprint，却被旧代码的
`missing:<candidate_id>` fallback 当作四个 singleton execution classes：

```text
mlp_fx_18e66507fb3d
mlp_fx_5b956b3eea29
mlp_fx_e1321f32002d
mlp_fx_e8035972ca13
```

根因是候选之间命中 compiler cache 时 Inductor 不重新发出 debug trace，而旧
`build_execution_classes()` 把缺失 hash 当成独立 class。该 session 必须永久保留为
diagnostic evidence，但不得进入 fingerprint 或 latency aggregation。registry 已设置：

```text
eligible_for_fingerprint_aggregation=false
eligible_for_latency_aggregation=false
```

不要删除或重新解释它。

## 7. Fingerprint/cache 修复

修复 commit：`76c6cd1287df59dbe0a86888c5271766fc6bd47e`

主要变更：

- 每个 compile candidate 使用 PyTorch 官方
  `torch.compiler.config.patch(force_disable_caches=True)`。
- session manifest 记录
  `candidate_compile_policy=force_disable_caches_per_candidate`。
- compile backend 要求每个 valid candidate 都有 v3 schema、artifact files、lowered hash、
  generated-code hash 和 execution hash。
- 任何 fingerprint 缺失都在 formal timing 前 hard-fail。
- `build_execution_classes()` 不再生成 `missing:<candidate_id>` singleton class。
- eager CPU runner 明确按 candidate profiling，不伪装成 execution-class dedup。

回归测试位于：

- `tests/unit/test_compile_candidate.py`
- `tests/unit/test_execution_classes.py`

## 8. 修复后的有效 MLP canary

有效 run：

```text
run_id: phase2_mlp_screening_v2_20260721_040600
session_id: mlp_d01_s01
artifact: .local-data/artifacts/phase2/phase2_mlp_screening_v2_20260721_040600/mlp_d01_s01
source: 76c6cd1287df59dbe0a86888c5271766fc6bd47e
elapsed: 46.219 seconds
status: ok
```

最终结果：

- 19 requested / 19 valid / 19 FX unique。
- 6 lowered unique / 6 execution unique。
- 全部 19 candidates 都有完整 `inductor-ir-v3` artifacts 和 hashes。
- formal timing 只包含 6 个 execution-class representatives。
- baseline class：`exec_241eedbb848ce50e`。
- point best class：`exec_d92a84b8651b02ad`。
- baseline-to-best gain：7.051%。
- best-worst spread：7.051%。
- monitor-off，timing sample_count=0。
- 无 foreign PID、无 contaminated rounds。
- boundary SM clock：1410 -> 1365 MHz。

execution mapping：

| execution class | candidates |
| --- | --- |
| `exec_0bf2a7f792dc1e22` | `mlp_fx_43017f8165ab`, `mlp_fx_55f081310d37`, `mlp_fx_5b52e166dbc3`, `mlp_fx_5b956b3eea29` |
| `exec_239a10e65d521e49` | `mlp_fx_1a631543e9f1`, `mlp_fx_c5489bfd6d92` |
| `exec_241eedbb848ce50e` | baseline `mlp_fx_b1d578ad8002`, `mlp_fx_99988e71f98c` |
| `exec_694a3e9839b6ac43` | `mlp_fx_47ca49eb22c7`, `mlp_fx_2b399a263e05`, `mlp_fx_6ee7991261c1`, `mlp_fx_0686d0c050f3`, `mlp_fx_e1321f32002d`, `mlp_fx_18e66507fb3d`, `mlp_fx_e8035972ca13` |
| `exec_92e6e1b2273d3e6d` | `mlp_fx_9fce48797b26` |
| `exec_d92a84b8651b02ad` | `mlp_fx_ee98f037e569`, `mlp_fx_e6bed09c25cc`, `mlp_fx_aabed146ca2b` |

这是完整 screening evidence，但还不是跨 session 结论。需要保留两个 replication
优先级信号：

- `exec_0bf2...` 的 same-class 短 diagnostic relative spread 为 2.554%，超过 2% floor。
- boundary clock 从 1410 降到 1365 MHz。

这两个信号不否定正式 6-class mapping，但使 `mlp_d01` 成为后续 adaptive replication
候选。不要把单 session 的 7.051% 当作稳定跨 context 结论。

## 9. 下一步执行顺序

### 9.1 先提交当前接力状态

确认有效 `mlp_d01_s01` registry 记录和本文件已经形成一个小 commit 并 push，保持
工作树 clean。不要提交 raw session、Inductor cache 或 generated code。

### 9.2 串行完成剩余 8 个 MLP screening groups

MLP 是 control family，不默认是主要学习空间。每组只运行一个 screening session，
每次运行前重新做 GPU gate，使用独立 Python process 和 cold cache。

剩余 groups：

```text
decode:
phase2_mlp_qwen_decode_bs2_t1_bf16
phase2_mlp_qwen_decode_bs4_t1_bf16
phase2_mlp_qwen_decode_bs8_t1_bf16
phase2_mlp_qwen_decode_bs16_t1_bf16

prefill:
phase2_mlp_qwen_prefill_bs1_s128_bf16
phase2_mlp_qwen_prefill_bs1_s512_bf16
phase2_mlp_qwen_prefill_bs1_s1024_bf16
phase2_mlp_qwen_prefill_bs1_s2048_bf16
```

decode 使用 `configs/profiling/phase2_discovery_decode_v1.json`，prefill 使用
`configs/profiling/phase2_discovery_prefill_v1.json`。不要在同一个 Python process
连续跑多个正式 sessions，也不要对 19 个 FX candidates 机械重复 formal profile；
runner 应只 profile execution-unique representatives。

每组结束立即审计：

- 19/valid/FX/lowered/execution counts
- 所有 valid candidates fingerprint 完整
- semantic plan 到 execution class 的多对一映射
- equivalence 和 alias
- formal profile unit 数等于 execution unique 数
- monitor-off 和 boundary snapshots
- GPU UUID、foreign PID、clock 和 contamination
- baseline-to-best gain、spread、CV 和 same-class warnings

任一组 fingerprint 或 mapping 不完整，停止扩展并调查；不要启用 fallback。

### 9.3 完成 MLP 离线聚合

现有 `scripts/analyze_phase2.py` 和 `phase2_analysis.py` 的核心 semantic/execution
口径可以复用，但 Markdown 和 `questions` 中仍有 RMSNorm 专用的 8 -> 6 文案与字段。
先泛化或加 MLP wrapper，不要直接生成带 RMSNorm 硬编码问题的 MLP 报告。

MLP 聚合必须报告：

- 19 FX 在各组形成多少 lowered/execution classes
- enumeration growth 与 execution growth 是否脱钩
- 每个 execution class 的 candidates 和 semantic plans
- strict/tie/ambiguous、noise-aware best set、winner entropy
- global/decode/prefill best fixed semantic plan
- fixed-plan P50/P90/max regret
- production/default 到 point/noise-aware oracle gap
- top-k oracle curves
- winner 是否随 decode/prefill、batch、sequence length 变化
- bs1/t1 的约 7% point gain 是否扩展到其他 workloads
- 是否一个固定 merged/fused semantic plan 已足够

### 9.4 Adaptive replication

不要机械复测全部 9 组。根据 screening 选择：

- 每种 provisional strict winner 的代表组
- gain 明显超过 2% 的组
- best-fixed tail regret 大的组
- decode/prefill winner 变化代表组
- CI 靠近 2% 的组
- clock/CV/same-class diagnostic 异常组
- fingerprint/mapping 可疑组

支撑主要结论的组补足至少 3 个独立完整 sessions。若一个 provisional winner 统治
全部 groups，至少复测小 decode、较大 decode 和最大或最有差异的 prefill。明确 tie
且 CI 支持 tie 的组不需要无限复测。

### 9.5 Phase 2 总决策

RMSNorm 和 MLP 完成后生成：

- `docs/reports/phase2/candidate_enumeration_report.md/json`
- `docs/reports/phase2/family_discovery_report.md/json`
- `docs/reports/phase2/phase2_exit_decision.md/json`

并更新 README 的最终 Conda/environment 配置和当前阶段状态。

Phase 2 exit 必须明确选择：

1. 进入正式数据集和强 baseline 阶段；
2. 转 benchmark/compiler diagnostic；
3. 淘汰当前 family，开始小规模 QKV/layout pilot；
4. 停止 learned selector 路线。

只有至少一个 family 同时具备真实 execution diversity、稳定超过 noise 的性能差异、
随 context 可复现变化的 winner/best set、非平凡 fixed/production tail regret 和足够
production-to-oracle gap，且现象不能由单一 fixed plan 或简单规则解决，才允许进入
下一阶段。

## 10. 最终汇报要求

完成后用中文汇报：

- final commit hash、工作树和 origin/main 状态
- 新增/修改文件和测试结果
- 所有 MLP screening 与 replication sessions
- invalid/aborted/contaminated sessions
- execution/semantic plan 映射
- strict/tie/ambiguous 统计
- fixed-plan regret 和 production-to-oracle gap
- RMSNorm family decision
- MLP control family decision
- Phase 2 exit direction
- README 环境配置更新

始终让实验结果决定下一步，不要以增加实验数量或维持 learned-selector 叙事为目标。
