# Phase 2 接手审计日志（2026-07-17）

## 审计范围

- 完整阅读源服务器外部环境说明与 docs/NEXT_CODEX_PROMPT.md。
- 完整阅读唯一研究总计划 docs/rewrite_research_plan.md。
- 阅读 README、artifact policy 和 docs/reports/phase1 下全部 Markdown/JSON。
- 核对项目结构、Git 状态、最近提交、origin、GitHub 登录与公开仓库状态。
- 未恢复 vertify、rewrite_miniexp 或旧 GNN 路线。

## 接手基线

- 接手时本地 HEAD 与 origin/main 均为 c8178cd。
- 工作树干净，origin 为 git@github.com:huamuyichun/rewrite.git。
- Phase 1 正式结论、execution class 映射和 monitor-off 协议与接力文档一致。
- 修改前基线为 16 passed，compileall 通过；MLP/RMSNorm 枚举数分别为 19/8。

## 本轮工程结果

提交 a96d20d（feat: 完成 Phase 2 多 family 适配）已经推送至 origin/main。

- 新增 FamilyAdapter，按 family_id 注入 workload 解析、baseline、candidate 和 input factory。
- runner 已支持 mlp_gate_up_activation_control 与 rmsnorm_residual_boundary。
- 统一枚举 CLI 与 runner 共用 enumerator registry，并检查配置 family 一致性。
- 新增 9 个 MLP 与 8 个 RMSNorm discovery groups。
- decode 配置 iterations_per_sample=20，prefill 配置为 1。
- 两套正式 discovery 配置均显式 monitor_mode=off，仅保留 NVML boundary snapshots。
- 新增统一 CLI、adapter 等价与两族 CPU runner 端到端回归。

验证结果：

- pytest：23 passed。
- compileall：通过。
- 12 个 JSON 配置全部可解析。
- 17 个 discovery group ID 全部唯一。
- MLP/RMSNorm CPU runner dry run 均完成 result.json 写盘。
- 推送后本地 HEAD 与 origin/main 均为 a96d20dc113ede559204acac63dda9287da2882b。

## GPU 入口审计

审计时间：2026-07-17 15:47:44 CST。

- GPU 0-6 均存在 foreign compute PID。
- GPU 利用率范围为 73%-100%。
- GPU 0-2 有其他用户 Python/VLLM 进程。
- GPU 3-6 被训练任务或 VLLM worker 占用。
- 没有发现正在运行的本项目 Phase 2 实验进程。

因此本轮没有选择 GPU，没有启动或排队任何 discovery session，也没有触碰其他用户进程。

## Session 与 artifact 状态

- 新增正式 GPU session：无。
- aborted session：无。
- contaminated session：无。
- 新增正式 artifact：无。
- artifacts/registry.jsonl：未修改。
- Phase 1 execution class 映射与 monitor policy：未修改。

## 下一步

GPU 重新空闲后，先同时复核 nvidia-smi 和 compute PID，只选择一张无 foreign PID 的卡。第一项建议运行单组 RMSNorm norm_only decode bs1/t1/bf16 canary；完成后先检查等价、lowering/execution collapse、污染与 boundary snapshots，再决定是否扩展，不机械跑完全部 17 组。
