# 给下一轮 Codex 的完整接力 Prompt

你正在接手服务器项目 /pub/data/hjwz/rewrite。不要从旧的 GNN 想法重新开始，也不要恢复已经删除的历史 pilot。当前唯一主计划是：

/pub/data/hjwz/rewrite/docs/rewrite_research_plan.md

项目目标是围绕现代 GPU LLM 推理编译，研究语义等价 rewrite candidates 在 FX、lowering 和最终 execution 层的塌缩、性能差异、噪声感知选择与受控测量。原始“用 GNN 选择 rewrite”已经被直接先例覆盖，不再作为方法主张。本阶段仍禁止训练任何学习模型。

## 零号优先级：先验证 GitHub 与本地同步

公开仓库 https://github.com/huamuyichun/rewrite 已创建，SSH origin 已绑定，main 已首次 push。新对话先验证 gh auth、origin、工作树和 origin/main 一致，再开始 Phase 2；不要重复创建仓库。

后续每个有意义的工程阶段都要形成 commit 并 push。大型 raw artifacts、cache、模型权重和 secrets 继续只保留本地。


## 一、接手后的第一组动作

先只做结构和状态审计，不要立刻启动 GPU 实验：

1. 完整阅读 /pub/data/hjwz/server_environment_prompt.txt。
2. 完整阅读 docs/rewrite_research_plan.md。
3. 阅读 README.md、docs/reports/phase1/ 下全部 md/json、docs/artifact_policy.md。
4. 执行 git status --short、git log --oneline -15、git diff --check。
5. 检查已提交的 Phase 2 adapter 与 discovery 配置，尤其是：
   - scripts/run_phase1_audit.py
   - scripts/enumerate_candidates.py
   - scripts/analyze_phase1.py
   - src/rewrite_selector/equivalence/validator.py
   - configs/profiling/phase1_monitor_policy_v1.json
   - configs/rewrites/rmsnorm_bounded_v1.json
6. 执行 CPU 测试：
   PYTHONPATH=src /pub/data/hjwz/miniconda3/envs/rewrite_miniexp/bin/python -m pytest -q
7. 执行语法检查：
   PYTHONPATH=src /pub/data/hjwz/miniconda3/envs/rewrite_miniexp/bin/python -m compileall -q src scripts tests
8. 确认工作树干净，拉取远端状态但不要覆盖本地历史。
9. 检查 Git/SSH/gh、origin 和公开仓库 huamuyichun/rewrite。
10. 验证 git remote -v、git status、当前 commit 和远端 main 一致。
11. 验证完成后，才确认实验进程与空闲 GPU，并且只使用一张明确空闲卡。

Phase 2 adapter 已在 a96d20d 完成；仍需先复核代码、测试和远端同步，再进入 GPU discovery。
2026-07-17 15:47 CST 的接手审计中没有任何明确空闲 GPU，因此未启动 session；后续必须重新检查实时状态。

## 二、服务器硬约束

- 活跃代码、缓存、日志和 artifacts 放在 /pub/data/hjwz，不要把高频 I/O 放到 /home/hjwz。
- Conda 环境：/pub/data/hjwz/miniconda3/envs/rewrite_miniexp。
- TMPDIR=/pub/data/hjwz/tmp。
- XDG_CACHE_HOME=/pub/data/hjwz/.cache。
- 用户允许自主执行非破坏性工程决策，不要反复询问。
- 删除操作仍需遵守 server_environment_prompt.txt；本轮已按用户明确要求删除旧 pilot，不要恢复。
- GPU 实验前必须同时检查 nvidia-smi GPU 状态和 compute PID。
- 只使用一张明确空闲 GPU，不排队、不抢占、不碰其他用户进程。
- 每个正式 session 使用独立 Python 进程和独立 cold Inductor cache。
- 不恢复 qwen_s03。
- 不训练 GNN、MLP、tree ranker、DeepSets 或其他 selector。
- 不做大规模采集，不超过约 40 个 discovery workload groups。
- 不运行完整 Chitu serving。
- 不根据实验结果修改 2% noise/tie floor 或 0.5% monitor self-effect gate。

## 三、已经完成并验证的工程

仓库已完成：

- 正规 Python/Git 项目结构、README、MIT License、CONTRIBUTING、pyproject、GitHub CI。
- .gitignore 排除了大型 raw artifacts、Inductor/Triton cache、模型权重、日志和 Python cache。
- 实验 registry：artifacts/registry.jsonl，schema v2。
- randomized blocked rounds。
- batched CUDA timing。
- independent session 和 cold cache metadata。
- 多 seed、多输入分布 eager/compiled equivalence。
- alias/in-place safety guard。
- FX、lowered、execution 三层 fingerprint。
- execution_class_id、canonical representative、candidate provenance。
- 正式 profiling 默认只测 execution-unique representative。
- 同 execution class candidate 的少量 diagnostic audit。
- NVML 环境/污染/clock 边界快照。
- Phase 1 跨 session strict/tie/ambiguous 分析。
- bounded MLP rewrite enumeration。
- RMSNorm/residual boundary family 的核心 IR 和 enumerator。

最近稳定提交：

- 87a2f8b 建立可复现基线。
- 9fa7022 registry schema v2。
- 6fea9f1 execution-class 分析。
- 526d77e registry dirty-state 修复。
- ad03259 bounded MLP enumerator。
- e747679 Phase 1 cross-session reporting。
- aa56a7b / a929755 monitor self-effect 工具。
- 3d76844 bounded RMSNorm family。
- e143b5d lowered fingerprint v2。
- e577642 Phase 1 provisional hold report。
- 3ceeb9c Phase 1 正式关闭与 family-discovery 基础设施。
- 0985e4a 旧 GNN/pilot 清理与完整接力文档。
- a96d20d 完成 MLP/RMSNorm family adapter、统一枚举回归和 17 组 discovery 配置。

上述内容均已提交并 push；当前下一步是等待一张明确空闲 GPU，先跑 RMSNorm decode 小 canary。

## 四、Phase 1 已经正式关闭

Phase 1 的核心结论已经成立：

1. 测量协议能够复现稳定性能差异。
2. 周期监控会扰动 timing，已经被识别并隔离。
3. 高层 candidates 会发生 lowering collapse。
4. tie-aware 标签是必要的。
5. 真实 Qwen2.5-7B decode bs1/t1/bf16 workload 存在约 6% 的稳定 baseline-to-best 空间。
6. 6 个 control candidates 不是搜索空间上限；bounded enumerator 已自然产生 19 个 FX candidates。
7. RMSNorm/residual boundary 已有第二 family 工程入口。

正式 Phase 1 报告：

- docs/reports/phase1/measurement_noise_report.md/json
- docs/reports/phase1/lowering_collapse_report.md/json
- docs/reports/phase1/fixed_baseline_report.md/json
- docs/reports/phase1/phase1_exit_decision.md/json

当前 phase1_exit_decision 应为 pass_phase1，七项 gate 全部 true，主统计 session 是 qwen_s06、qwen_s07、qwen_s08。

## 五、Qwen session 清单

固定 workload：

- Qwen2.5-7B MLP
- hidden_dim=3584
- intermediate_dim=18944
- bf16
- decode
- batch=1
- token/seq_len=1
- 单 A100

Session：

- qwen_s01：failed，初始 bf16 tolerance 不合适，禁止汇总。
- qwen_s02：旧版完整有效 session，约 4.59% preliminary gain。
- qwen_s03：aborted/incomplete，只完成 5 个 candidate audit，永远禁止恢复或汇总。
- qwen_s04：完整、async NVML、6 FX / 6 旧 lowered / 4 execution、gain 约 6.05%。
- qwen_s05：完整、async NVML、gain 约 6.09%。
- qwen_s06：正式 monitor-off 协议，6 FX / 4 lowered / 4 execution、gain 6.108%。
- qwen_s07：正式 monitor-off 协议，gain 6.023%。
- qwen_s08：正式 monitor-off 协议，gain 6.092%。

关键 artifact：

- artifacts/phase1/phase1_qwen_decode_20260716/qwen_s02
- artifacts/phase1/phase1_qwen_decode_20260716/qwen_s03
- artifacts/phase1/phase1_qwen_decode_repeat_20260717/qwen_s04
- artifacts/phase1/phase1_qwen_decode_repeat_20260717/qwen_s05
- artifacts/phase1/phase1_qwen_decode_monitor_off_20260717/qwen_s06
- artifacts/phase1/phase1_qwen_decode_monitor_off_20260717/qwen_s07
- artifacts/phase1/phase1_qwen_decode_monitor_off_20260717/qwen_s08

所有 s06-s08 都是独立进程、cold cache、source_dirty=false、0 contaminated rounds。

## 六、execution class 与 fingerprint 结论

lowered fingerprint v1 错误地把 fx_graph_transformed.py 中的源码行号、局部变量名和 session 路径纳入 hash，导致 fused candidates 看似跨 session 不稳定。

e143b5d 将 lowered fingerprint v2 限定为：

- ir_pre_fusion.txt
- ir_post_fusion.txt

并保留 generated code / execution fingerprint 单独分析。修复后稳定结果为：

- 6 FX fingerprints
- 4 lowered fingerprints
- 4 execution fingerprints

已知 execution class：

- exec_241eedbb848ce50e：p0_baseline_separate_silu + p2_separate_inplace_silu_mul
- exec_239a10e65d521e49：p1_separate_manual_silu
- exec_694a3e9839b6ac43：p3_fused_chunk_silu + p4_fused_split_silu_inplace
- exec_0bf2a7f792dc1e22：p5_fused_chunk_manual_silu

正式赢家是 exec_694a3e9839b6ac43。p3/p4 必须作为同一个 execution class。p3/p4 class 与 p5 class 在锁定的 2% noise floor 下是稳定 tie。candidate-level结果仅用于诊断；oracle、best fixed、win share 和 regret 必须按 execution class 计算。

## 七、monitor self-effect 结论

重复创建 nvidia-smi 子进程曾导致 SM clock 从 1410 MHz 降到 765 MHz，已确认不可用。

NVML 避免子进程，但周期查询仍没有稳定通过 0.5% gate：

- 1 Hz 长批次：median absolute paired delta 约 0.899%。
- 0.5 Hz：约 0.546%。
- 0.333 Hz：约 0.597%。
- 5 秒轮询的一个 run 虽为 0.311%，但 timing window 内 sample_count=0，因此无效，不能作为通过证据。
- 所有最终 clean runs clock 均为 1410 MHz，无 foreign PID，无 contaminated phase。

选定正式策略：

- timing window 内 monitor_mode=off。
- 只在 blocked timing 前后做 NVML boundary snapshots。
- 周期 monitor verdict=failed_disabled。
- 配置：configs/profiling/phase1_monitor_policy_v1.json。

关键 monitor artifacts：

- artifacts/phase1/phase1_monitor_self_effect_clean_long_20260717/result.json
- artifacts/phase1/phase1_monitor_self_effect_clean_2s_5cycle_20260717/result.json
- artifacts/phase1/phase1_monitor_self_effect_clean_3s_5cycle_20260717/result.json

不要把结论写成“async monitor 无干扰”。正确表述是“周期监控未通过 gate，所以已从正式 timing 隔离；正式协议本身不并发轮询”。

## 八、bounded MLP enumerator

文件：

- src/rewrite_selector/rewrites/mlp_enumerator.py
- src/rewrite_selector/ir/mlp.py
- configs/rewrites/mlp_bounded_v1.json
- scripts/enumerate_candidates.py

默认：

- max_rewrite_depth=3
- max_fx_unique_candidates=32
- 支持调整到 64
- 稳定 BFS + hash 排序
- growth：depth 0/1/2/3 大致为 1/4/10/19
- 19 enumerated/valid/FX-unique candidates
- 保存全部 provenance traces、invalid applications、rule hypotheses、enumeration tree

规则轴：

- separate/merged gate-up projection
- gate_up/up_gate packing
- chunk/split/narrow
- native/decomposed SiLU
- out-of-place/safe in-place multiply

不要为了达到 32 强行增加 identity、变量名、无意义 reshape 或其他语法噪声。下一阶段必须先 FX dedup，再 lowered/execution dedup，只正式 profile execution-unique candidates。

## 九、RMSNorm family

已提交核心文件：

- src/rewrite_selector/ir/rmsnorm.py
- src/rewrite_selector/rewrites/rmsnorm_enumerator.py
- tests/unit/test_rmsnorm_enumerator.py
- configs/rewrites/rmsnorm_bounded_v1.json 当前未提交

family_id=rmsnorm_residual_boundary。

已实现候选轴：

- native RMSNorm
- square_mul decomposition
- square_pow decomposition
- flatten/restore hidden rows
- scale reassociation
- norm_only context
- residual_silu producer context

它被选择是因为语义容易严格验证、可以程序枚举、有机会影响 Inductor fusion/materialization，并且不是切换手写 kernel。

## 十、Phase 2 adapter 与 discovery 准备状态

Phase 2 adapter 已在 a96d20d 完成并推送。当前功能状态：

1. scripts/enumerate_candidates.py 通过共用 registry 分派 mlp_bounded / rmsnorm_bounded。
2. validator 通过可注入 input_factory 支持两族输入。
3. runner 根据 rewrite_config.family_id 选择 Workload/RMSNormWorkload。
4. baseline、candidate instantiate 与 input factory 均由 FamilyAdapter 注入。
5. MLP control 的 Phase 1 路径保持兼容。
6. 已加入两族统一枚举 CLI、adapter 等价和 CPU runner 端到端回归。
7. 已加入 9 个 MLP 与 8 个 RMSNorm discovery groups。
8. decode/prefill 使用不同 iterations_per_sample，正式 timing 均为 monitor_mode=off。
9. 旧 GNN/pilot 删除和主计划引用修复保持不变，不要恢复。

a96d20d 验证结果：

- pytest：23 passed。
- compileall：通过。
- 12 个 JSON 配置解析通过。
- MLP enumeration CLI：19 candidates，growth 1/4/10/19。
- RMSNorm enumeration CLI：8 candidates，growth 1/4/8。
- 两族 CPU runner dry run 均写出完整 result.json。

下一优先级是等待一张明确空闲 GPU，先启动单组 RMSNorm decode canary。不要排队，不要覆盖 registry，不要恢复 qwen_s03。

## 十一、旧路线清理决定

用户已明确决定：

- 不再考虑原始“GNN rewrite selector”方法。
- vertify 是旧 GNN 时代手工最小闭环，已删除。
- rewrite_miniexp 和相关旧 eager pilot 已删除。
- mainline_task.md、verified_execution_roadmap.md、旧续聊 prompt、旧指标草稿已删除。
- docs/rewrite_research_plan.md 是唯一主计划。

仍保留计划中的 GNN/先例讨论，只用于解释为什么原始 claim 不成立、为什么当前阶段禁止直接训练模型。不要重新创建旧目录或把旧结果混入正式数据。

## 十二、Phase 2 当前执行顺序

以下准备工作已经完成：family adapter、统一 enumeration CLI 回归、discovery workload/profiling configs、CPU enumeration/equivalence dry run、提交与 push。

下一步：

1. 同时检查 nvidia-smi 状态和 compute PID。
2. 只有一张卡无 foreign PID 且利用率明确空闲时，才启动单组 RMSNorm decode canary。
3. canary 完成后先审计 equivalence、collapse、execution fingerprints、污染和 monitor boundary snapshots。
4. 只有 canary 干净，才继续剩余 RMSNorm groups，再决定是否运行 MLP control groups。
5. 每轮先看 collapse、noise、winner variation 和 regret，不机械跑完 17 组。

建议 discovery groups，总计 17：

MLP control，9 groups：

- decode：batch=1,2,4,8,16，seq_len=1，bf16，Qwen dims
- prefill：batch=1，seq_len=128,512,1024,2048，bf16，Qwen dims

RMSNorm/residual pilot，8 groups：

- norm_only decode：batch=1,8，seq_len=1
- residual_silu decode：batch=1,8，seq_len=1
- norm_only prefill：seq_len=128,1024
- residual_silu prefill：seq_len=128,1024
- hidden_dim=3584，bf16

根据显存和时间可缩减，但总量保持 6-12 个 RMSNorm groups、整个 discovery 不超过约 40 groups。decode 与 prefill 使用不同 iterations_per_sample 配置，避免 prefill 2048 被 decode 的 batch timing 放大。

正式 profiling：

- monitor_mode=off
- NVML boundary snapshots
- randomized blocked rounds
- execution-class representatives only
- same-class 少量 fingerprint audit
- 不因某个 candidate 更快而提前停止其他 class
- 保存完整 provenance、rounds、raw samples、environment manifest

## 十三、Phase 2 必须计算

按 family 和 execution class：

- enumerated count
- valid count
- FX unique count
- lowered unique count
- execution unique count
- lowering/execution retention
- collapse classes
- 每条 rule 对 execution diversity 的边际贡献
- noise-aware spread
- strict/tie/ambiguous 比例
- execution-class win share
- winner entropy
- global/per-family best fixed regret
- production/default regret
- P50/P90/max regret
- production-to-oracle gain
- top-k oracle
- prefill/decode 分层结果

重点回答：

1. 是否一个 execution class 统治绝大多数 groups。
2. winner 是否只由简单 shape threshold 决定。
3. winner 是否与 context/layout/fanout 等图信息相关。
4. candidate 增加是否真的带来 execution diversity。
5. 是否大量候选只发生 lowering collapse。

最终生成：

- docs/reports/phase2/candidate_enumeration_report.md/json
- docs/reports/phase2/family_discovery_report.md/json
- docs/reports/phase2/phase2_entry_decision.md/json

phase2_entry_decision 三选一：

1. 继续 learned selector。
2. 转 benchmark/compiler diagnostic。
3. 停止当前 candidate family，继续寻找新 family。

在看到足够 execution diversity、稳定 winner variation、非平凡 fixed-plan tail regret 和 production-to-oracle gap 之前，不得进入学习模型阶段。

## 十四、GitHub 状态

目标仓库：https://github.com/huamuyichun/rewrite

GitHub 配置已完成：

- gh 已登录 huamuyichun。
- origin 为 git@github.com:huamuyichun/rewrite.git。
- 仓库 visibility=PUBLIC，default branch=main。
- 本地 main 跟踪 origin/main。
- 首次 push 已包含提交 0985e4a 及之前完整历史。


不要上传：

- artifacts 下大型 raw traces/caches
- 模型权重
- Inductor/Triton cache
- 中断 session 二进制
- secret、SSH key、token

后续工作必须保持小步 commit，并在验证通过后 push origin/main。

## 十五、实验反馈原则

用户要求实验结果必须反馈到下一步设计，不是机械执行计划。

已经发生的有效反馈包括：

- nvidia-smi 干扰 clock，替换为 NVML。
- NVML 周期查询仍未通过 gate，因此彻底移出 timing window。
- lowered fingerprint 被源码注释污染，因此升级为 IR-only v2。
- 六个 candidate 在执行层塌缩为四类，因此正式统计改为 execution class。
- 单 session 4.59% 只算 preliminary，三次 monitor-off clean sessions 后才确认约 6%。
- 手工六候选只做 control，扩展为 bounded 19-candidate enumeration。
- MLP 可能被 merged plan 统治，因此增加 context-sensitive RMSNorm family。

后续也必须如此：每一轮 discovery 先看 collapse、noise、winner variation 和 regret，再决定规则、family 或研究方向，不要为了 candidate 数量或 learned selector 叙事制造结果。

## 十六、交付要求

下一轮完成工作时必须汇报：

- commit hash
- GitHub remote/push 状态
- 修改/删除文件
- 测试结果
- 完整 session 列表
- aborted/contaminated session
- artifact 路径
- execution class 映射
- monitor policy
- enumeration 接口
- discovery 结论
- 是否允许进入学习模型阶段

始终以这个核心目标判断工作是否有价值：

找到多个语义合法、lowering 后执行不同、性能差异超过噪声，并且会随真实 workload/context 可复现地交换优劣的 execution classes。
