# Rewrite 项目研究与执行计划

- 更新日期：2026-07-17
- 文档状态：当前主计划
- 目标窗口：面向 2026 年秋季投稿窗口，但以阶段门槛而不是日历驱动
- 适用范围：Git 仓库根目录（服务器迁移见 `docs/SERVER_MIGRATION_GUIDE.md`）

> 本文档是唯一主计划。被其取代的旧路线和手工 pilot 已在 Phase 1 完成后删除；control seeds 已迁移到版本化配置。
>
> 执行状态：Phase 0/1 已完成，正式结论见 `docs/reports/phase1/`；第 1 节保留的是立项时反证背景，不代表当前工程状态。

## 0. 结论先行

### 0.1 当前结论

项目可以继续，但必须重定位。原来的核心表述：

> 用 GNN/图级 latency estimator 在等价 rewrite plans 中选择最快计划。

不能再作为新颖性主张。已有工作已经覆盖了其中的大部分组合：

- Kaufman 等人在 MLSys 2021 已用 GraphSAGE 和真实 TPU 测量做 XLA operator-fusion 配置的 learned performance model，并接入 autotuner 减少硬件测量。
- X-RLflow 在 MLSys 2023 已用 GNN + PPO 逐步选择 tensor graph rewrites，并使用稀疏的端到端 latency 反馈。
- TpuGraphs 在 NeurIPS 2023 已公开“同一图、多配置、真实 runtime、group-wise ranking”的大规模数据和 GraphSAGE/ranking baseline。
- 2025 年的 configuration cross-attention 工作已在 TpuGraphs 上使用候选集合条件编码和 pairwise ranking，说明“GNN + pairwise ranking”本身也不是方法创新。
- TENSAT、TASO、vLLM 和本地 Chitu 都已经包含 gate/up projection 合并和 fused SiLU-and-mul 一类优化。当前 6 个 candidate 的主要 rewrite 不是新规则。

因此，项目推荐改为：

> **面向现代 GPU LLM 推理编译的、噪声感知的等价 rewrite 候选相对排序与选择性测量。**

更具体地说：在 PyTorch FX/Inductor 上构造真实 LLM block 的等价 rewrite candidate groups，学习候选之间的相对性能与不确定性；高置信度时直接选 plan，低置信度时只 lowering/编译/profile 少量候选，以更少优化成本接近 oracle 或 `max-autotune`。

### 0.2 可以主张与不能主张的内容

可以争取的贡献：

1. 一个公开、可复现的 GPU/LLM/Inductor 等价 rewrite benchmark，标签来自真实编译后执行，而不是 FLOPs 或单算子求和。
2. 一个显式面向 candidate set 的相对排序与不确定性模型，服务 fixed-budget selection，而不是只做绝对 latency 回归。
3. 一个多保真决策策略：高层 FX 预筛选、少量 lowering 后重排、极少量真实 profile，并分别量化节省的编译和测量成本。
4. 对前端图差异、lowering 后执行差异、测量噪声和 selector regret 之间关系的系统分析。

不能再主张：

- 首次用 GNN 做图优化或 rewrite 选择。
- 首次用 learned cost model 做 fusion/configuration selection。
- 首次用 pairwise ranking 预测 tensor program 性能。
- gate/up fusion、manual SiLU、in-place multiply 本身是新 rewrite。
- 当前 3 个 block 已证明图模型必要。
- eager 的 60-graph 结果等价于真实 compiler rewrite 结果。

### 0.3 总体可行性

| 维度 | 判断 | 原因 |
| --- | --- | --- |
| 工程闭环 | 高 | 已有 FX 抽图、candidate 实例化、Inductor profiling 和专用 Conda 环境 |
| 数据构造 | 中 | 工具链可用，但当前不是 rewrite engine，且需要真实 workload、lowered fingerprint 和严格等价验证 |
| 测量可信度 | 中低 | 当前只运行一次、candidate 固定顺序、样本少、共享服务器噪声未完整建模 |
| selection 问题成立 | 尚未成立 | 当前固定选择 `p3` 的 median regret 为 0，最大 regret 仅约 2.38%，简单规则几乎做完 |
| GNN 必要性 | 无证据 | 当前只有同一拓扑、6 个固定 plan 和 3 个正式 workload group |
| 研究新颖性 | 有条件 | 原始命题高度重合；必须依靠 GPU/LLM/Inductor benchmark、多保真选择和 uncertainty 形成差异 |
| ICLR 适配度 | 高风险 | 如果最终只是系统 profiling + GNN，会偏系统工程；需要清楚的学习问题、OOD 结果和方法增量 |

当前决策是“有条件继续”，而不是直接进入模型阶段。
### 0.4 Phase 1 执行进展补充（2026-07-17）

本节是在原计划基础上的执行记录，不替换后续阶段门槛。Phase 0 与 Phase 1 已完成，正式报告位于 docs/reports/phase1/。

已经完成的工程与实验：

1. 建立了正规 Git/Python 项目结构、版本化配置、CI、测试、artifact policy 和 experiment registry。
2. profiler 已升级为 randomized blocked rounds、batched CUDA timing、独立进程 session、cold cache 记录、环境 manifest 和污染检测。
3. 等价验证覆盖多个 seed、normal/uniform/zeros/extremes 输入、eager/compiled 输出和 alias/in-place guard。
4. 建立 FX、lowered IR、execution 三层 fingerprint，并将正式统计单位提升为 execution class。
5. bounded MLP enumerator 已取代“六个手工 candidate 是空间上限”的假设；默认 max depth=3、budget=32，自然产生 19 个 FX-unique candidates。
6. 第二个 context-sensitive family 已选择 RMSNorm/residual boundary，并完成核心 IR、规则枚举器与 CPU 等价测试入口。

Phase 1 的正式 Qwen canary 为 Qwen2.5-7B MLP，hidden=3584、intermediate=18944、bf16、decode、batch=1、token=1、单 A100。主统计使用三个独立 cold-cache、monitor-off session：qwen_s06、qwen_s07、qwen_s08。三次 baseline-to-best gain 分别约为 6.108%、6.023%、6.092%，winner execution class 一致，contaminated round ratio 均为 0。

fingerprint 审计发现并修复了一项真实测量问题：旧 lowered hash 把 transformed-FX 中的源码行号、局部变量名和 session 路径当作 compiler 差异。Phase 1 正式报告使用的 lowered fingerprint v2 只使用 ir_pre_fusion.txt 与 ir_post_fusion.txt。服务器迁移后代码升级为 v3，额外归一化动态 workspace/cache 路径；v2 历史映射为：

- 6 个 FX fingerprints。
- 4 个 lowered fingerprints。
- 4 个 execution fingerprints。
- p0/p2 塌缩为同一 execution class。
- p3/p4 塌缩为同一 execution class。
- p3/p4 class 与 p5 class 在锁定的 2% noise floor 下为稳定 tie。
- 正式赢家为 p3/p4 对应的 fused native-SiLU execution class。

monitor self-effect audit 同样改变了正式协议。同步 nvidia-smi 曾导致 SM clock 从 1410 MHz 降至 765 MHz；改用 NVML 后不再创建子进程，但周期 NVML 在 1 Hz、0.5 Hz 和约 0.333 Hz 下的 median absolute paired effect 仍分别约为 0.899%、0.546% 和 0.597%，没有通过锁定的 0.5% gate。因此周期监控被标记为 failed_disabled：正式 timing window 内 monitor_mode=off，只在 blocked timing 前后采集 NVML boundary snapshots 以检查 clock 和 foreign PID。不能将该结论写成“async monitor 无干扰”。

Phase 1 exit decision 的七项 gate 已全部通过：

- profiler 可复现。
- 正式 instrumentation 不在 timing 内并发轮询。
- 三层 fingerprint 与 execution-class mapping 跨 session 稳定。
- strict pair 排序可复现。
- p3/p4/p5 的 tie/collapse 结论稳定。
- Qwen baseline-to-best gain 跨 session 存在。
- 四个 execution classes 中存在稳定排序。

因此允许进入小规模 Phase 2 candidate-family discovery，但仍不允许训练学习模型。Phase 2 必须先完成当前 MLP/RMSNorm 共用 runner adapter，再在不超过约 40 个 groups 的范围内检查 execution diversity、winner variation、fixed-plan tail regret 和 production-to-oracle gap。

## 1. 已有内容审计

### 1.1 已完成资产

当前主线已经从早期手工 pilot 迁移到可复现实验框架。

早期最小闭环曾完成：

1. 锁定 PyTorch eager/Inductor、Torch FX、SwiGLU-like MLP 和单卡范围。
2. 抽取 MLP FX graph，保留 op、shape、dtype 和 dependency。
3. 定义 6 个 control seed plans。
4. 实例化候选并做初步等价与 FX 去重。
5. 完成初步 eager/Inductor profiling。

原始 pilot 目录已删除，不进入正式统计。当前证据入口为：

- [`reports/phase1/phase1_exit_decision.md`](reports/phase1/phase1_exit_decision.md)
- [`reports/phase1/lowering_collapse_report.md`](reports/phase1/lowering_collapse_report.md)
- [`../artifacts/registry.jsonl`](../artifacts/registry.jsonl)
- [`../configs/rewrites/mlp_control_v1.json`](../configs/rewrites/mlp_control_v1.json)

环境资产：

- 源服务器历史 Conda 环境：Python 3.12.13；目标服务器路径不固定
- Python：3.12.13
- PyTorch：2.10.0+cu129
- CUDA runtime：12.9
- GPU：A100 40GB/80GB 多卡共享服务器
- 本地真实模型：Qwen2.5-7B-Instruct，`hidden=3584`、`intermediate=18944`、`bf16`
- 源服务器历史 Chitu（未纳入仓库）：提交 `d5cbf84`

### 1.2 正式 Inductor pilot 能说明什么

正式口径只有 3 个 workload group、每组 6 个 candidates：

| workload | baseline p50 | oracle | oracle p50 | spread | winner flip |
| --- | ---: | --- | ---: | ---: | --- |
| seq128 h1024 i4096 | 0.096768 ms | p5 | 0.086016 ms | 12.50% | false |
| seq128 h768 i3072 | 0.097280 ms | p3 | 0.089600 ms | 12.57% | false |
| seq512 h768 i3072 | 0.108544 ms | p3 | 0.104448 ms | 3.92% | true |

这些结果只支持：

- 6 个高层 FX graph 可以执行，并在当前 tolerance 下通过单输入等价检查。
- 高层 FX signature 不重复。
- 在 2 个 workload 上，Inductor 后仍存在约 12.5% 的候选 latency spread。
- 第 3 个 workload 的差异已经接近噪声，winner 不稳定。

它们不支持：

- candidate 在 Inductor kernel/codegen 层仍有 6 种独立实现。
- 图结构比 plan ID 或 shape rule 更有预测价值。
- selector 能泛化到未见 block、rewrite family 或模型。
- 当前空间需要学习器。

### 1.3 当前最关键的反证

在 3 个正式 workload 上，固定选择 `p3_fused_chunk_silu`：

- median regret：0
- max regret：约 2.38%
- 在 2 个 workload 上等于 oracle，在另 1 个 workload 上为第二名

这意味着当前 candidate space 几乎被“总是选 fused chunk SiLU”做完。继续在这 18 条 summary row 上训练 XGBoost、MLP 或 GNN 没有研究意义。

此外，当前 baseline 是 separate gate/up projection，而 vLLM 和本地 Chitu 已经把 merged gate/up + fused SiLU-and-mul 作为生产实现路径之一。用弱 baseline 得到的 oracle gain 不能代表相对现代 serving baseline 的真实收益。

### 1.4 60-graph eager 实验的定位

早期 eager 扩展实验曾包含 60 个 graph ID、46 个唯一 workload shape、6 个 candidates，但：

- 所有 graph 都是同一 MLP 拓扑和同一 rewrite family。
- 它没有接真实 compiler pass manager，标签是 eager CUDA latency。
- 改变随机权重或 seed 不会形成新的结构样本，也不能视为独立泛化单位。
- heuristic 中含有手工的 sequence-length 分段规则，不应当作未调参的强证据。
- candidate 以固定顺序连续测量，可能与 GPU drift、cache 和温度变化混杂。

因此它只保留为早期 profiling 工具原型，不进入最终论文训练集和主结果。

### 1.5 当前实现缺口

1. 项目目录还不是 Git repository，缺少版本历史和实验 commit 绑定。
2. 没有根级 README、`pyproject.toml`/lockfile、统一 CLI、单元测试和 schema version。
3. candidate 是手工 module 参数组合，不是可扩展的 rewrite rule/enumerator。
4. 等价验证只有单个随机输入，`atol=2e-3`、`rtol=2e-2`，没有边界输入、多个 seed、alias/in-place 检查或编译后输出检查。
5. 去重只看 FX signature，没有 AOT/Inductor IR、generated code、kernel sequence 去重。
6. profiling 每个 candidate 连续测完，顺序固定；没有 randomized blocked rounds 和跨进程重复。
7. winner flip 只比较同一次 run 的前后半段，不等价于独立 session 的排序复现率。
8. 没有 GPU 温度、功耗、时钟、其他进程、driver、compiler flags 等完整环境指纹。
9. 数据划分、label tie policy、noise floor、ranking metric 尚未固化。
10. 当前 shape `h=768/1024`、dtype=fp16，与本地 Qwen2.5-7B 的 `h=3584/i=18944/bf16` 不一致，也没有 decode regime。

## 2. 相关工作与新颖性边界

### 2.1 最接近的公开工作

| 工作 | 已覆盖内容 | 对本项目的含义 | 可复用性 |
| --- | --- | --- | --- |
| [TASO, SOSP 2019](https://doi.org/10.1145/3341301.3359630) | 自动生成 tensor graph substitutions，cost-based search | rewrite space/search 不是新问题 | 可借 rewrite 规则和 search baseline；直接接 PyTorch 成本高 |
| [TENSAT, MLSys 2021](https://arxiv.org/abs/2101.01332) | equality saturation + tensor graph extraction；包含共享输入 matmul 合并类 multi-pattern rule | 当前 gate/up fusion 规则已有直接先例 | [代码](https://github.com/uwplse/tensat)可参考规则、等价和 extraction |
| [A Learned Performance Model for TPUs, MLSys 2021](https://arxiv.org/abs/2008.01040) | GraphSAGE、真实 TPU runtime、tile-size pairwise ranking、operator-fusion prediction、autotuner top candidate 测量 | 与“图级 learned cost model + fusion selection + 减少 profile”高度重合 | 无完整公开 corpus/code；模型和评估必须作为主要 baseline |
| [Transferable Graph Optimizers, 2020](https://arxiv.org/abs/2010.12438) | GNN + RL 联合做 placement/fusion/scheduling 并讨论未见图迁移 | “GNN 优化 ML compiler graph”不是新意 | fusion/scheduling 使用分析模型，真实 placement 与本项目口径不同 |
| [DNNFusion, PLDI 2021](https://doi.org/10.1145/3453483.3454083) | graph rewrite、fusion plan generation、轻量 profiling | 强非学习 fusion baseline 不能只用 fusion count | 可借 memory/launch/resource 特征设计 |
| [X-RLflow, MLSys 2023](https://arxiv.org/abs/2304.14698) | GNN + PPO 逐步选择 rewrite，候选图编码，稀疏端到端 latency reward | 与“GNN 选 rewrite”直接重合；论文还指出高层 cost model 与 E2E 可偏 24% | [代码](https://github.com/ucamrl/xrlflow)为 MIT，但 CUDA 10.2/TASO 老栈且不能泛化未见图 |
| [TpuGraphs, NeurIPS 2023](https://openreview.net/forum?id=plAix1NxhU) | 约 31.1M layout pairs + 12.87M tile pairs，同图多配置、真实 runtime、graph holdout、top-K slowdown | benchmark schema、group split、ranking loss、fixed-budget metric 均有直接模板 | [数据与 baseline](https://github.com/google-research-datasets/tpu_graphs)可直接参考，目标平台不同 |
| [Graph neural networks with configuration cross-attention, 2025](https://arxiv.org/abs/2405.16623) | GraphSAGE + cross-configuration attention + pairwise hinge，只做组内排序 | “candidate-set encoder + pairwise GNN ranking”也已有先例 | 未找到官方代码；必须实现等价强 baseline |
| [Tensor E-graphs + MCTS, PACT 2024](https://doi.org/10.1145/3656019.3689611) | MCTS 构建 e-graph，快速 runtime extraction | 搜索/提取是替代技术路线 | [代码](https://github.com/jakobhartmann/tensor-eqs-mcts)可参考 cost/extraction |
| [PerfSAGE, 2023](https://arxiv.org/abs/2301.10999) | GNN 预测任意 TFLite DNN 的 latency/energy/memory | 图级性能预测本身已有充分先例 | 用于检验 GNN 是否真比无图聚合强 |
| [TLP, ASPLOS 2023](https://doi.org/10.1145/3575693.3575737) | schedule primitive sequence cost model，集成 Ansor，跨硬件迁移 | sequence model 是强 baseline，不可只比普通 MLP | [代码](https://github.com/zhaiyi000/tlp)可参考 tokenization/ranking |
| [CDMPP, EuroSys 2024](https://arxiv.org/abs/2311.09690) | compact AST + domain adaptation，跨模型/跨设备性能预测 | OOD 与少量目标数据适配是重要 baseline | [代码](https://github.com/joapolarbear/cdmpp)可参考迁移评估 |
| [NeuSight, ASPLOS 2025](https://arxiv.org/abs/2407.13853) | tile/kernel-aware 的分析与学习混合 GPU 性能模型 | 纯高层图模型可能信息不足 | [代码](https://github.com/scai-tech/NeuSight)可参考 hybrid 特征 |

还应纳入强系统基线：

- [FusionStitching](https://arxiv.org/abs/2009.10924)
- [AStitch / BladeDISC](https://doi.org/10.1145/3503222.3507723)
- [Welder](https://www.usenix.org/conference/osdi23/presentation/shi)
- PyTorch Inductor default 与 `mode="max-autotune"`
- vLLM 的 `MergedColumnParallelLinear + SiluAndMul`
- 本地 Chitu 的 `FeedForwardHFLlama.merge_gate_up` 和 `silu_and_mul`

### 2.2 调研后的明确判断

答案不是“还没人做”，而是：

> **核心想法已经有人做过，而且有多个直接先例；尚未发现完全相同的公开工作把现代 PyTorch Inductor、真实 LLM rewrite candidate groups、组内相对 ranking、uncertainty/abstention 和分级编译/profile 预算放在同一个开源研究闭环中。**

“尚未发现”不是排他性证明。进入投稿写作前必须再做一次更新检索和 citation chaining。

### 2.3 最可信的差异化轴

优先级从高到低：

1. **多保真决策，而非单一 predictor**：FX 级全量预筛、少量 lowering 后重排、极少量真测。
2. **噪声和不确定性感知**：候选差异接近 noise floor 时允许 tie/abstain，而不是强行生成错误 winner label。
3. **现代开放 GPU 编译栈**：PyTorch 2.10/Inductor/Triton/A100，可公开复现。
4. **真实 LLM workload regime**：prefill + decode、真实 model dimensions、bf16/fp16，至少一个 context-sensitive family。
5. **相对/差分表示**：显式编码 baseline graph、candidate graph 和 rewrite delta，而非独立预测绝对 latency。
6. **公开 benchmark 和 protocol**：数据按 candidate group、原始图和 rewrite family 防泄漏划分。

“LLM-specific”单独不够，因为 TpuGraphs、X-RLflow 和后续 GPU performance model 都已经覆盖 Transformer/LLM 相关 workload。

## 3. 修订后的研究问题

### 3.1 问题定义

一个样本组定义为：

`q = (G0, P, c, v)`

其中：

- `G0`：原始高层计算图或被 rewrite 的作用域图。
- `P={p1...pn}`：语义等价 candidate plans。
- `Gp`：应用 plan `p` 后的 candidate graph。
- `c`：执行条件，包括模型维度、prefill/decode、token 数、batch、dtype、layout、GPU。
- `v`：编译器和运行时指纹，包括 PyTorch/Inductor/Triton/CUDA/driver 版本及关键 flags。
- `L(q,p)`：真实编译后稳态 latency 的统计分布，不是单次测量值。

Stage-A 模型：

`sA(G0, Gp, delta_p, c, v) -> score, uncertainty`

只使用 compile/lowering 前可得信息，对所有 candidates 进行低成本排序。

Stage-B 模型：

`sB(G0, Gp, lowered_p, c, v) -> score, uncertainty`

只对 Stage-A 选出的少量 candidates 做 lowering/编译，使用 kernel graph、generated code metadata 等信息重排。

最终策略：

1. 高置信度时只编译 Stage-A top-1。
2. 中等置信度时编译/lower top-m，再选 Stage-B top-1。
3. 低置信度时只真实 profile Stage-B top-k，选实测最优。

目标不是最小化 RMSE，而是在给定编译和测量预算 `B` 下最小化：

`selected_latency + lambda_compile * compile_cost + lambda_profile * profile_cost`

论文必须分别报告 candidate 总数、lowered/compiled 数、profiled 数和总优化 wall time，避免把使用 lowered IR 的模型错误描述为“compile 前选择”。

### 3.2 研究问题

RQ1：在现代 Inductor 上，哪些高层等价 rewrite 在 lowering 后仍产生可区分的执行计划和稳定 latency 差异？

RQ2：真实 LLM shape、prefill/decode 和局部上下文是否会改变 winner，使固定规则或 production default 无法接近 oracle？

RQ3：组内相对/差分模型能否在 held-out shape、model 和 rewrite context 上优于强分析、tree、sequence 和 GraphSAGE baseline？

RQ4：uncertainty/abstention 能否识别 near-tie 和 OOD group，并以少量额外 lowering/profile 降低 tail regret？

RQ5：Stage-A + Stage-B + selective profile 能否以显著少于全量 autotune 的成本接近 oracle/`max-autotune`？

### 3.3 可证伪假设

H1：至少一个 rewrite family 在大多数 workload group 上的 candidate spread 显著高于 measurement noise，并且 lowering 后不完全塌缩。

H2：production default 和 best fixed plan 在 held-out groups 上有非平凡 regret，winner 随 context 发生可复现变化。

H3：candidate-delta 或 set-aware 模型在 decision metrics 上稳定优于不看结构的强 baseline，而不是只降低 regression error。

H4：校准不确定性与真实 ranking error 正相关，选择性测量能改善 coverage-risk 和 fixed-budget regret。

任何一条失败，都必须触发收缩或转向，不能通过换更重模型规避。

## 4. 研究边界

### 4.1 主线范围

- 单机、单 GPU 执行；数据可在不同空闲 A100 上分批采集，但每条 run 绑定具体 device fingerprint。
- 主 IR：Torch FX。
- 主 backend：`torch.compile(backend="inductor")`。
- 主硬件：A100，先固定一种显存/型号口径作为主数据。
- 主 workload：dense Transformer block，必须覆盖真实 prefill 和 decode regime。
- 主 dtype：与真实模型一致的 bf16；fp16 作为条件或补充，不混成同一无条件任务。
- 主任务：candidate group ranking + uncertainty + fixed-budget selection。
- Chitu 只用于最后的生产路径验证，不作为前期 rewrite engine。

### 4.2 明确不做

- 不做任意全模型 e-graph 搜索。
- 不训练 RL sequential rewrite policy，与 X-RLflow 正面重复。
- 不把 kernel schedule tuning、serving scheduler、KV cache policy 混入主任务。
- 不在第一主线做多机、多节点或在线请求调度。
- 不以跨 GPU 架构泛化作为必须 claim；没有 H100 数据就不声称跨代泛化。
- 不同时支持多个 compiler backend。
- 不从零实现复杂 GNN；先复用 GraphSAGE/set-aware baseline。
- 不把随机权重或重复 profile 当成新的独立 graph sample。

### 4.3 投稿分支

方法分支：如果 set-aware delta + uncertainty/multi-fidelity 在 OOD 和 fixed-budget 上有稳定优势，可面向 ICLR/MLSys 的 learning-for-systems 叙事。

benchmark/system 分支：如果模型方法增量有限，但公开数据、protocol、lowering 分析和系统 budget 结果扎实，更适合 MLSys/ASPLOS/CGO 类 venue。

止损分支：如果 production baseline 已接近 oracle、candidate 经 lowering 大量塌缩或差异低于噪声，则停止“learned selector”主张，转为 compiler diagnostic/benchmark，或更换 rewrite family。

## 5. Candidate 与 workload 设计

### 5.1 Candidate family 选择原则

一个 family 进入主数据集前必须同时满足：

1. 规则有清楚的语义前提和可自动验证的适用条件。
2. candidate 可以程序化枚举，不依赖人工为每个 shape 写 module。
3. 高层图有差异，且有足够比例的 candidate 在 lowering/kernel 层仍不同。
4. winner 会随 shape/context 变化，而不是一个 fixed plan 长期统治。
5. 相对 production baseline 存在可用 oracle gain。
6. 不把“选不同手写 kernel”伪装为 graph rewrite。

### 5.2 推荐的 family pilot

Family A：MLP gate/up + activation 边界。

- 保留现有 6 candidates 作为 control family。
- baseline 改为同时报告 HF/Inductor default、vLLM/Chitu-style merged gate-up。
- 增加真实 bf16、decode token 数和 Qwen/Llama/Mistral dimensions。
- 目标是验证“该 family 是否已经被 always-merge 做完”，不是把它当主创新。

Family B：QKV projection 与 layout/view/transpose propagation。

- separate Q/K/V 与合法的 merged QKV projection。
- split/chunk/narrow/view 的不同表达。
- reshape/transpose/view/contiguous 的合法移动与消除。
- 必须固定 attention kernel 边界，避免把 FlashAttention/SDPA kernel choice 混为 rewrite。

Family C：RMSNorm/residual/pointwise fusion boundary。

- 只允许有严格数据依赖和别名条件的等价 regroup。
- 关注 producer/consumer context 对 Inductor fusion boundary 的影响。
- 不改变 pre-norm/post-norm 语义。

优先做 A 作为 control，B/C 各做小 pilot。最终主数据只保留通过门槛的 1-2 个 family，不要求三个都做大。

### 5.3 Workload 轴

独立 workload group 至少由以下轴共同定义：

- model family/config：Qwen2.5、Llama/Mistral 风格及一个 held-out config。
- phase：decode 与 prefill 分开。
- batch/token regime：decode 使用真实小 token/batch；prefill 使用多档 sequence length。
- hidden/intermediate/head/KV-head dimensions。
- dtype/quantization：主线先 bf16；量化只在后续作为单独 domain，不混入首轮。
- local graph context：rewrite 作用域的 producer/consumer、layout、fanout、alias 条件。
- compiler configuration/version。

当前 `seq=128/512` 的 MLP microbenchmark 只能代表部分 prefill，不能代表 decode 或完整 LLM inference。

### 5.4 数据规模目标

数据规模以 candidate group 数而不是 row 数计：

- discovery set：约 30-50 个独立 workload groups，用于筛 family 和测 noise。
- baseline set：至少数百个独立 groups，每组保留多个 lowering 后独立 candidates。
- model set：目标 500-1000+ 独立 groups、数千到数万 candidate rows；最终规模由 profiling 成本和 learning curve 决定。

这些是容量目标，不是为了凑数。若 learning curve 已饱和或 group 不独立，应停止扩数据。重复 seed、重复测量和相同 shape 的不同随机权重不能增加 group 数。

## 6. Rewrite 与等价性基础设施

### 6.1 Rewrite 表示

每条 rule 必须有：

- `rule_id` 和 schema version。
- pattern/replacement。
- shape、dtype、layout、fanout、alias 等 preconditions。
- 参数变换，例如 merged weight 的 concat 维度。
- 数学等价说明和 floating-point caveat。
- 可逆性或 provenance。
- 适用 block/family。

candidate plan 必须记录有序 `rewrite_trace`，而不是只记录最终 plan 名称。

### 6.2 等价验证

等价性采用分层验证：

1. 静态检查：shape/dtype、边数量、alias/precondition、参数映射。
2. 多输入数值检查：多个 seed、正态/均匀/零/极值和不同合法 shape。
3. 高精度参考：能够使用 fp32/fp64 的子表达式先与高精度 reference 比较。
4. 编译前与编译后分别检查输出。
5. 对 in-place candidate 检查 use-count 和 alias，不允许修改仍被使用的 tensor。
6. 对可能积累数值差异的候选，增加真实 block/model accuracy sanity check。

near-equal 不等于语义等价。超出锁定 tolerance 或存在模型回归的 candidate 直接标记 invalid，不参与性能 oracle。

本地 Chitu 已对部分模型强制 `silu_and_mul_impl="torch"` 以规避稳定性问题，这说明数值 guard 是主流程要求，不是附加测试。

### 6.3 三层去重

1. High-level：规范化 FX graph + constant/attribute/shape fingerprint。
2. Lowered：AOT/Inductor IR、fusion group、generated kernel/code fingerprint。
3. Execution：kernel name/type、launch count、grid/block、tensor shape/stride sequence fingerprint。

只有 FX 不同但 lowered/execution 相同的 candidates：

- 保留在“lowering collapse”分析中。
- 不重复 profile 和计入有效 candidate 数。
- 不能作为模型可分性的证据。

## 7. Profiling 与标签协议

### 7.1 测量目标

主标签是稳态 block latency 分布。以下成本分开记录：

- FX rewrite/enumeration time。
- lowering/compile time。
- first-run/prime time。
- steady-state execution latency。
- profile 次数和总 profile wall time。
- selector inference time。

不要把 compile time 混入 execution latency，也不要在声称节省 compile 时忽略 Stage-B 已产生的 compile 成本。

### 7.2 推荐测量流程

1. 每个 run 使用独立 `run_id`、config 和环境 manifest。
2. 运行前后记录 GPU 型号/UUID、其他进程、utilization、温度、功耗、时钟、driver 和 CUDA/PyTorch 版本。
3. candidate 顺序按 round 随机化；每个 candidate 在本 round 先重新 warmup，再测一个连续 batch，兼顾稳态和 drift 控制。
4. 至少跨多个独立 process/session 复测，不用单次 run 的前后半段替代复现。
5. 使用 CUDA Event + synchronize，保留每次 raw sample。
6. 对微秒级 workload 使用足够重复或 batched iteration，避免计时量化占主导。
7. 编译 cache 策略必须固定并显式区分 cold/warm cache。
8. 使用 bootstrap confidence interval 和相对差异，而不只看 CV。
9. 对 near-tie 自适应增加测量；明显劣势 candidate 提前停止，减少共享 GPU 时间。
10. profiling 期间如果检测到外部 GPU process/utilization 变化，标记 contaminated 并重排，不静默纳入标签。

### 7.3 Tie 和噪声标签

先从独立重复估计 workload-specific noise floor `epsilon_q`。

候选 `pi` 与 `pj` 只有在相对差异的置信区间稳定越过 `epsilon_q` 时才生成严格 preference；否则标为 tie/ambiguous。

训练和评估均需支持：

- strict pair。
- tie pair。
- invalid candidate。
- collapsed candidate。

不允许为每组强制指定唯一 winner。当前 b2 的 p3/p5 应更接近 tie，而不是把一次 p50 的微小差异当 ground truth。

### 7.4 共享服务器约束

- 默认只占用一张明确空闲的 GPU，设置 `CUDA_VISIBLE_DEVICES`。
- 不启动多卡并行采集，不常驻占卡，不抢占已有进程。
- 先运行短 canary，再提交可中断的小批次任务。
- 不锁全局 GPU clocks，不修改影响其他用户的 driver/system 配置。
- 使用低优先级或服务器现有调度机制；每个 batch 可断点续跑。
- 大规模采集前估算 candidate 数、预计 compile/profile 时长和输出空间。
- 原始数据即时落盘，失败后只补缺失 group，不整批重跑。

## 8. 数据 schema 与防泄漏

### 8.1 最小 schema

每个 group/candidate 至少保存：

- `dataset_version/run_id/group_id/candidate_id`。
- source model/block/phase/workload shape。
- baseline graph、candidate graph、rewrite trace。
- high-level/lowered/execution fingerprints。
- node/edge/global features。
- equivalence status 与误差统计。
- raw latency samples、summary、confidence interval、noise floor。
- compile/prime/profile costs。
- GPU/compiler/environment manifest。
- validity/collapse/contamination flags。

### 8.2 划分原则

所有 candidates of one group 必须在同一 split。禁止 candidate-row random split。

至少报告：

1. shape interpolation：同 family/model 下 held-out shape。
2. shape extrapolation：held-out token/hidden bucket。
3. model/config holdout：held-out model dimensions。
4. context holdout：相同 rule 在未见 producer/consumer context。
5. family holdout：只作为 stress test；如果模型不承诺跨 rule family 泛化，不作为主 claim。
6. version shift：有条件时用 compiler patch/version 变化做 robustness 分析。

group、canonical graph、相同 lowered fingerprint 和近重复 shape 必须做 lineage 检查，避免跨 split 泄漏。

### 8.3 禁止特征

- 任意 `candidate_id`、plan 名称或人为排序序号。
- 从真实 latency 派生的统计量。
- primary Stage-A 模型中使用 lowered/kernel 特征。
- 会直接泄漏 winner 的手调规则输出。

## 9. Baseline 体系

Baseline 必须按由弱到强排列，不能只比较 GNN 与普通 MLP。

### 9.1 决策下界与上界

- production/default plan。
- random plan，多 seed。
- global best fixed plan。
- per-family best fixed plan。
- shape-bucket lookup / nearest-neighbor。
- oracle、noise-aware oracle 和 top-k oracle。

`global best fixed plan` 是当前最重要 baseline；现有 p3 结果表明它可能已经接近 oracle。

### 9.2 规则与分析 baseline

- op/kernel/launch count。
- FLOPs、parameter bytes、activation bytes、estimated memory traffic。
- roofline/critical-path proxy。
- op-wise microbenchmark sum，模拟 TASO-style cost。
- fusion boundary、fanout、layout/stride/contiguous penalty。
- FusionStitching/DNNFusion/Welder 启发的 resource/memory 特征。
- Inductor default。
- Inductor `max-autotune`，同时报告 compile/profile cost。

kernel count 等 post-lowering 特征只能作为 Stage-B 或 diagnostic upper-bound baseline，不能混入 Stage-A。

### 9.3 非图学习 baseline

- linear/logistic pairwise model。
- XGBoost/LightGBM/HistGradientBoosting，包含 pairwise/group ranking 版本。
- MLP/DeepSets on bag-of-ops + rewrite metadata。
- TLP-style linearized graph/rewrite sequence encoder。

### 9.4 图与集合 baseline

- Kaufman/TpuGraphs-style GraphSAGE + global reduction。
- GraphSAGE + candidate config late/early join。
- cross-configuration attention + pairwise hinge/listwise loss。
- candidate 独立编码与 set-aware 编码消融。

### 9.5 多保真 baseline

- Stage-A top-k 后直接 profile。
- Stage-A top-m + Stage-B analytical rerank。
- 全量 compile、全量 profile。
- random/heuristic top-k profile。

## 10. 推荐模型路线

### 10.1 先做相对/差分特征模型

第一学习模型不是 GNN，而是：

`score = f(summary(G0), summary(Gp), delta(G0,Gp), context)`

用 tree ranker 或小 MLP 验证：

- plan delta 是否有信息。
- winner 是否可跨 shape 泛化。
- simple baseline gap 是否真实存在。

### 10.2 Set-aware delta ranker

只有上一阶段通过后，主模型采用：

1. 共享 encoder 编码 baseline 和 candidates。
2. 构造 `candidate - baseline` 与显式 rewrite delta。
3. 用 candidate-set attention 建模组内相对关系。
4. 条件向量编码 workload、hardware/compiler fingerprint。
5. 输出 score 和 aleatoric/epistemic uncertainty。

图 encoder 可替换为 GraphSAGE、DAG message passing 或 linearized encoder。模型贡献应来自 delta/set/multi-fidelity/uncertainty，而不是“用了 GNN”。

### 10.3 Loss

主 loss 候选：

- tie-aware pairwise logistic/hinge loss。
- ListMLE/ListNet 或 differentiable top-k surrogate。
- delta-latency regression 作为辅助。
- uncertainty calibration loss。

pair sampling 按 group 和显著性分层，避免大量 near-tie 或同一大 group 支配训练。

### 10.4 Multi-fidelity 与 teacher-student

推荐实现顺序：

1. Stage-A student：只看 FX/pre-lowering features。
2. Stage-B teacher/reranker：看 lowered/kernel graph features。
3. 比较 Stage-A、Stage-B 和联合 policy 的预算曲线。
4. 如果 Stage-B 明显强，可将 teacher score/embedding 蒸馏到 Stage-A；蒸馏不是首轮必做。

这能直接回答“高层信息是否足够”。如果 Stage-A 在 OOD 上失败而 Stage-B 成功，应诚实把系统定位为 compile-budget reduction，而不是 compile-free selection。

## 11. 评价协议

### 11.1 主决策指标

- selected-plan latency。
- regret：`L_selected / L_oracle - 1`。
- oracle gain capture：`(L_default - L_selected) / (L_default - L_oracle)`。
- top-k slowdown error：模型 top-k 中实测最佳相对 oracle 的 slowdown。
- fixed-budget selected latency。
- profile/compile/lowering count reduction。
- optimization wall time 和 amortization break-even。
- coverage-risk curve：只对高置信度 group 自动决策时的 coverage 与 regret。
- P50/P90/P99 regret，不只报平均值。

### 11.2 辅助指标

- pairwise accuracy，strict/tie 分开。
- Kendall tau / Spearman。
- NDCG/top-k hit rate。
- latency delta MAE/MAPE，仅作辅助。
- uncertainty calibration：ECE、Brier/NLL、error-detection AUROC 或 risk-coverage AUC。

### 11.3 统计要求

- 模型至少多 seed，报告均值和置信区间。
- group-level bootstrap，不对 candidate rows 独立 bootstrap。
- 与最强 baseline 做 paired test。
- 主结果按 family、phase、shape bucket 分层。
- 明确报告 invalid、collapsed、ambiguous 和 contaminated group 比例。
- 所有阈值在 validation set 固定，不根据 test 结果调整。

## 12. 阶段计划与验收门槛

计划不按天拆分。每个阶段只有通过 exit gate 才进入下一阶段。

### Phase 0：课题重定位与工程基线

任务：

1. 与导师确认修订后的贡献边界和直接先例，特别是 Kaufman、X-RLflow、TpuGraphs、2025 cross-config ranking。
2. 将早期 pilot 排除出正式 dataset，并迁移仍有用的 control seeds。
3. 初始化版本控制，建立 root README、环境锁、配置规范、实验 registry 和测试目录。
4. 固化问题定义、指标、split policy 和禁止 claim。
5. 建 literature matrix，记录论文任务、输入层级、标签、split、metric、代码和差异。

产出：

- canonical README 和本计划。
- versioned experiment registry 与 Phase 1 reports。
- environment lock 和 repo skeleton。
- related-work matrix。

Exit gate：导师接受“原始 GNN claim 不新，主线改为 noise-aware multi-fidelity selection”。若不能接受，应先换题，不继续堆实验。

### Phase 1：测量、等价与 lowering 审计

任务：

1. 重写 profiler 为 randomized blocked rounds + independent sessions。
2. 增加环境监控、污染检测、cold/warm cache 口径。
3. 增加多输入、alias、编译后等价验证。
4. 抓取 Inductor debug/codegen/kernel artifacts，建立三层 fingerprint。
5. 在现有 3 groups 上复测并量化 `p3` fixed baseline、tie 和 collapse。
6. 用真实 Qwen MLP shape 做小 canary，不立即扩大数据。

产出：

- versioned profiling protocol。
- noise report。
- lowering-collapse report。
- equivalence test suite。

建议 Exit gate，首次审计后锁定具体数值：

- 独立 session 的 pairwise order/winner 有可接受复现率。
- latency CI 宽度明显小于计划研究的 effect size。
- contaminated run 可自动识别。
- lowered fingerprint 能稳定复现。

若失败：先修测量，不进入 candidate expansion 或模型。

### Phase 2：Candidate family discovery

任务：

1. 把现有 MLP family 作为 control，换 production baseline。
2. 分别做 QKV-layout 和 RMSNorm/residual 小 pilot。
3. 覆盖真实 bf16、prefill/decode、真实 hidden/intermediate/head dimensions。
4. 每个 family 同步跑 fixed plan、shape lookup、analytical baseline。
5. 统计 high-level unique、lowered unique、winner entropy、oracle gain 和 noise-aware spread。

产出：

- family comparison matrix。
- 被拒绝 family 及原因。
- 最终保留的 1-2 个 family 和规则说明。

建议 Exit gate：

- 大部分有效 group 的 spread 至少约为 noise floor 的 2 倍。
- lowering 后仍保留足够的独立 candidates，而非多数塌缩。
- 没有单一 plan/简单规则在几乎所有 group 上统治。
- best fixed plan 在 tail 上有非平凡 regret。
- production baseline 到 oracle 有足够可兑现收益。

如果 MLP family 仍由 merged plan 统治，将它降为 control；如果所有 family 都失败，停止 learned selector 路线。

### Phase 3：正式数据与数据质量

任务：

1. 程序化枚举 groups/candidates，建立断点续跑和 content-addressed cache。
2. 使用自适应测量减少共享 GPU 时间。
3. 固化 schema、lineage、split 和 dataset card。
4. 按 model/phase/context 平衡采样，不只扫规则网格。
5. 生成 learning curve，决定是否继续扩到更大规模。

产出：

- dataset v1。
- raw/profile/summary/lowered artifacts。
- data quality dashboard/report。
- frozen train/val/test manifests。

Exit gate：

- 数百个以上真正独立 groups，且不靠 seed/重复 shape 凑数。
- invalid/collapse/noisy 比例可解释。
- split 无 lineage 泄漏。
- production/fixed baseline gap 在 test 设计前已经由 discovery set 证明存在。

### Phase 4：强 baseline 决策点

任务：

1. 完成 default、production、fixed、lookup、analytical、op-sum、Inductor max-autotune。
2. 完成 tree ranker、MLP/DeepSets、TLP-style sequence。
3. 完成 Kaufman/TpuGraphs GraphSAGE 和 cross-config attention baseline。
4. 统一输出 fixed-budget 和 tail-regret 表。

产出：

- baseline leaderboard。
- error/failure slice。
- 是否需要新模型的书面结论。

Exit gate：

- 若 production/fixed/tree baseline 已接近 noise-aware oracle，转 benchmark/system 分支，不上更重模型。
- 只有结构或 candidate-set 信息仍有明确 gap，才进入 Phase 5。

### Phase 5：Set-aware delta 与多保真策略

任务：

1. 实现 baseline/candidate/delta encoder。
2. 实现 tie-aware set ranking。
3. 加 uncertainty calibration 与 abstention。
4. 实现 Stage-A/Stage-B/selective profile policy。
5. 视结果决定是否做 teacher-student distillation。

产出：

- 主模型和 checkpoint。
- budget-policy implementation。
- ablation：无 delta、无 set、无 graph、无 uncertainty、无 Stage-B。

Exit gate：

- 在预注册主 split 上稳定优于最强 baseline，而非只赢 weak MLP。
- 提升反映在 selected latency/P90 regret/budget curve。
- uncertainty 能识别 near-tie/OOD error。
- selector 开销相对节省的 compile/profile 成本可忽略或可摊销。

若失败：保留数据和 baseline，转 benchmark/system 分支。

### Phase 6：主实验与真实系统验证

任务：

1. 完成 shape/model/context OOD 和 learning curve。
2. 完成 full-budget curve：0 profile、少量 compile、少量 profile、全量 autotune。
3. 在本地 Chitu 或 production-like block 中验证候选选择是否仍带来 layer/model 级收益。
4. 比较 Inductor default、max-autotune、项目 policy 和 oracle。
5. 分析高层预测失败但 lowering 后成功的 case。
6. 检查 microbenchmark gain 是否在端到端被稀释。

产出：

- 主结果表和图。
- OOD/ablation/failure-case report。
- Chitu/production-like validation。
- optimization cost 和 break-even 分析。

Exit gate：主要 claim 在独立 test 和真实系统验证中均成立；若只在 microbenchmark 成立，必须收紧论文标题与贡献。

### Phase 7：论文与发布

任务：

1. 冻结数据、代码、环境、配置和结果 commit。
2. 清理一键复现实验，发布小数据样例与完整数据说明。
3. 写清直接先例和差异，不使用“首次 GNN rewrite selector”表述。
4. 将主图围绕 budget-regret、multi-fidelity 和 lowering collapse 组织。
5. 完成 artifact checklist、limitations、shared-server measurement caveat。

产出：

- 论文稿。
- dataset/model card。
- reproducibility package。
- release notes。

## 13. 建议的固定决策阈值

以下不是论文结果目标，而是防止无限投入的建议门槛。Phase 1 结束后根据实测 noise floor 一次性锁定，之后不能随 test 结果修改。

### 13.1 Candidate-space gate

- 至少 70% 的主数据 group，best-worst spread 大于约 `2 x noise floor`。
- lowering 后有效 candidate retention 建议不低于 40%-50%。
- 单一 fixed plan 的 win share 不宜超过约 70%。
- best fixed plan 的 P90 regret 建议至少达到 3%，否则学习选择的 tail 价值很弱。
- production baseline 到 oracle 的 median gain 建议至少有 3%-5%，并在一部分 group 上更高。

### 13.2 Selector gate

- 在受限预算下，median selected latency 接近 oracle，且 P90 regret 明显优于 strongest baseline。
- 以 20%-30% candidate profile budget 达到全量 profile 的大部分收益。
- 相对 strongest baseline 的提升在多 seed、多个 split slice 上方向一致。
- OOD group 可通过 uncertainty 降低自动 coverage，而不是自信地选错。

如果真实 noise 或 production gain 不支持这些数量级，应调整阈值并收紧 claim，而不是挑选 workload。

## 14. 风险、触发条件与转向

| 风险 | 早期信号 | 动作 |
| --- | --- | --- |
| 原始 novelty 已被覆盖 | Kaufman/X-RLflow/TpuGraphs/TGraph 可直接回答 claim | 使用修订定位；不能解决则换题 |
| always-merged 规则做完 | p3/production plan 几乎总胜 | MLP 降为 control，换 context-sensitive family |
| lowering 抹平候选 | FX unique 很高，kernel fingerprint retention 很低 | 提前 dedup；换 family，不换模型 |
| 测量噪声大 | CI 接近 spread，跨 session winner 不稳 | tie label、自适应测量、修 protocol |
| 共享服务器污染 | 外部 process/utilization 与 latency 同时变化 | 小批次、污染检测、重排；不锁时钟抢资源 |
| 数据泄漏 | random row split 结果异常高 | group/model/context split + lineage audit |
| 简单模型已经做完 | fixed/tree/lookup regret 接近 oracle | 转 benchmark/system 分支 |
| 高层信息不足 | Stage-A 失败、Stage-B 明显成功 | 定位为 compile-budget reduction；可做 teacher-student |
| 新 op/compiler version OOD | 未见 opcode 或 version shift 误差激增 | unknown-op encoding、few-shot calibration、限制 claim |
| 数值等价不稳 | faster candidate 有 block/model 输出回归 | invalid，禁止进入 oracle；加强 accuracy guard |
| microbenchmark gain 消失 | block 提升但 layer/model 无提升 | 收紧为 block benchmark，或停止系统 claim |
| 模型过重 | selector time/compile savings 不成比例 | tree/small encoder 优先，压缩模型 |
| Inductor 私有 API 变化 | debug/lowered capture 随版本破坏 | pin version、adapter layer、保存原始 artifacts |
| ICLR 叙事偏工程 | 方法消融无增量，贡献主要是数据管线 | 改投 systems-for-ML venue |

## 15. 工程目录建议

不立即搬动历史结果。新代码按下面结构建设：

```text
rewrite/
  README.md
  pyproject.toml
  configs/
    workloads/
    rewrites/
    profiling/
    experiments/
  src/rewrite_selector/
    ir/
    rewrites/
    equivalence/
    lowering/
    profiling/
    data/
    baselines/
    models/
    policies/
    evaluation/
  tests/
    unit/
    integration/
  scripts/
  docs/
  artifacts/            # 大结果，默认不入 Git
```

最低测试要求：

- rewrite precondition/parameter mapping 单测。
- 多输入等价和 alias 单测。
- schema validation。
- fingerprint 稳定性。
- group split 无泄漏。
- regret/top-k/tie metric 单测。
- 一个 CPU dry run 和一个可选 GPU integration smoke。

## 16. 最小可发表单元

投稿前必须形成下面这个闭环，而不是只训练一个 GNN：

1. 至少 1 个真正 context-sensitive 的 rewrite family，加 1 个 MLP control family。
2. 数百个以上独立 candidate groups，覆盖真实 LLM prefill/decode 和 model dimensions。
3. 真实 Inductor latency、完整 noise/tie protocol 和 lowering/kernel fingerprint。
4. production/default/fixed/analytical/tree/sequence/GraphSAGE/set-aware 强 baseline。
5. 一个有增量的 set-aware delta + uncertainty 或 multi-fidelity policy。
6. fixed-budget compile/profile 曲线和 tail regret。
7. 至少一次 Chitu 或 production-like layer/model 验证。
8. 公开可复现代码、schema、数据说明和限制。

如果只能完成第 1-4 项，成果定位为 benchmark/negative result；如果第 5-7 项也成立，才具备方法论文主线。

## 17. 接下来最先做的三件事

1. **先梳理并完成当前 Phase 2 runner WIP。** 保证 MLP/RMSNorm 共用 equivalence、fingerprint、execution-class 和 profiling 口径。
2. **完成受控 family discovery。** 用真实 Qwen bf16 prefill/decode shape，快速淘汰 always-win 或 lowering-collapse 的 family，总量不超过约 40 groups。
3. **生成 Phase 2 三份正式报告。** 依据 execution diversity、winner variation、fixed regret 和 production-to-oracle gap 决定下一路线。

在 Phase 2 决策前，不训练学习模型，不跑大规模数据，不接 Chitu 主流程。

## 18. 参考链接

- [TASO](https://doi.org/10.1145/3341301.3359630)
- [TENSAT](https://arxiv.org/abs/2101.01332)
- [A Learned Performance Model for Tensor Processing Units](https://arxiv.org/abs/2008.01040)
- [Transferable Graph Optimizers for ML Compilers](https://arxiv.org/abs/2010.12438)
- [DNNFusion](https://doi.org/10.1145/3453483.3454083)
- [X-RLflow](https://arxiv.org/abs/2304.14698)
- [TpuGraphs](https://openreview.net/forum?id=plAix1NxhU)
- [Graph neural networks with configuration cross-attention for tensor compilers](https://arxiv.org/abs/2405.16623)
- [Optimizing Tensor Computation Graphs with Equality Saturation and MCTS](https://doi.org/10.1145/3656019.3689611)
- [PerfSAGE](https://arxiv.org/abs/2301.10999)
- [TLP](https://doi.org/10.1145/3575693.3575737)
- [CDMPP](https://arxiv.org/abs/2311.09690)
- [NeuSight](https://arxiv.org/abs/2407.13853)
- [FusionStitching](https://arxiv.org/abs/2009.10924)
- [AStitch](https://doi.org/10.1145/3503222.3507723)
- [Welder](https://www.usenix.org/conference/osdi23/presentation/shi)
- [PyTorch Benchmark Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [X-RLflow code](https://github.com/ucamrl/xrlflow)
- [TpuGraphs data/baselines](https://github.com/google-research-datasets/tpu_graphs)
- [tensor-eqs-mcts code](https://github.com/jakobhartmann/tensor-eqs-mcts)
- [vLLM Llama implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py)

