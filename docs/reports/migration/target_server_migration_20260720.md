# 目标服务器迁移日志（2026-07-20）

## 结论

目标服务器 `bear` 已按 `docs/SERVER_MIGRATION_GUIDE.md` 完成阶段 A-I。
CPU 门禁、单卡 A100 GPU 门禁、GPU smoke 和三个独立 cold-cache Phase 1
recalibration session 均通过。目标端跨 session 分析结果为 `pass_phase1`，
允许在本迁移提交形成并 push 后进入 Phase 2 RMSNorm 单组 discovery。

迁移期间未恢复旧 GNN/pilot，未训练 selector，未修改锁定的 2% noise/tie
floor 或 0.5% monitor self-effect gate。

## Git 与主机

- 迁移基线 commit：`375e68df9f968baf64f4781b2ba11526f8509ebc`。
- `main` 与 `origin/main` 同步；`git pull --ff-only origin main` 为 already up to date。
- origin：`git@github.com:huamuyichun/rewrite.git`；GitHub SSH 认证成功。
- `gh` CLI 未登录，但不影响 Git SSH fetch/pull/push。
- 主机：`bear`，Ubuntu 24.04.1 LTS，Linux 6.8.0-134-generic，x86_64。
- `/Data1` 为只读 XFS 且已用 96%，未写入。数据根选择本机 ext4 上的
  `/home/hejwz/rewrite/.local-data`；安装环境后该文件系统约剩 36 GB。

目标路径变量：

```text
REWRITE_ROOT=/home/hejwz/rewrite
LOCAL_DATA_ROOT=/home/hejwz/rewrite/.local-data
REWRITE_ARTIFACT_ROOT=/home/hejwz/rewrite/.local-data/artifacts
REWRITE_REGISTRY_PATH=/home/hejwz/rewrite/artifacts/registry.jsonl
QWEN_MODEL_DIR=/home/hejwz/rewrite/.local-data/models/Qwen2.5-7B-Instruct
TMPDIR=/home/hejwz/rewrite/.local-data/tmp
XDG_CACHE_HOME=/home/hejwz/rewrite/.local-data/cache
HF_HOME=/home/hejwz/rewrite/.local-data/cache/huggingface
TORCH_HOME=/home/hejwz/rewrite/.local-data/cache/torch
TRITON_CACHE_DIR=/home/hejwz/rewrite/.local-data/cache/triton
CUDA_DEVICE_ORDER=PCI_BUS_ID
```

## Python 与 GPU 环境

- Conda prefix：`$LOCAL_DATA_ROOT/envs/rewrite`。
- Python：3.12.13（conda-forge）。
- PyTorch：2.10.0+cu129；CUDA runtime：12.9；Triton：3.6.0。
- NVIDIA driver：560.35.05（源服务器记录为 575.57.08）。
- 正式迁移卡：NVIDIA A100 80GB PCIe，物理 index 0。
- GPU UUID：`GPU-b0f2d831-6a1a-c820-f388-431148eabf25`。
- `pip check`：通过。

本机默认 CUDA ordinal 与 `nvidia-smi` index 的排序曾不一致。设置
`CUDA_DEVICE_ORDER=PCI_BUS_ID` 后，index 0/1/2 分别对应 A100/H100/A800。
迁移自检已增加 PyTorch UUID 与 NVML UUID 一致性检查，防止混合 GPU
服务器上选错物理卡。

## Qwen 与历史 raw artifacts

- Qwen 权重未下载，immutable revision 未获取；`QWEN_MODEL_DIR` 缺少
  `config.json`，迁移自检按设计报告 warning。
- Phase 1 recalibration 和 Phase 2 microbenchmark 只使用 Qwen 维度与随机初始化
  参数，不读取真实权重，因此该缺失不阻塞当前阶段。Chitu/production-like
  验证前仍须下载并执行 `scripts/check_migration.py --require-qwen`。
- 未同步源服务器 Phase 1 raw artifacts；目标 artifact root 中没有 qwen_s06-s08
  raw session，checksum 不适用。Git 中的 compact registry 与正式 Phase 1 报告
  作为历史证据保留。

## CPU 门禁

- `scripts/check_migration.py`：必需项通过；Qwen/raw artifacts/GPU（CPU 模式）为 warning。
- `python -m compileall -q src scripts tests`：通过。
- `PYTHONPATH=src python -m pytest -q`：28 passed；只有未安装可选 NumPy 的
  PyTorch import warning。
- MLP enumeration：19 candidates，growth 1/4/10/19，未截断。
- RMSNorm enumeration：8 candidates，growth 1/4/8，未截断。
- `git diff --check`：通过。

## GPU smoke

- session：`migration_gpu_smoke_20260720_0247/smoke_s01`。
- artifact：`$REWRITE_ARTIFACT_ROOT/migration/migration_gpu_smoke_20260720_0247/smoke_s01`。
- source commit：`375e68d`，`source_dirty_at_run=false`，独立 cold cache。
- 6/6 candidates 通过 eager/compiled equivalence 和 alias guard。
- unique count：6 FX、4 lowered、4 execution；fingerprint schema 全部为
  `inductor-ir-v3`，lowered/execution hash 非空。
- formal timing 为 monitor-off，sample_count=0；前后 A100 boundary clock 均为
  1410 MHz；无 foreign PID，`contaminated=false`。
- smoke latency 不进入 Phase 1/2 汇总。

## Phase 1 target recalibration

共同 run id：`phase1_target_recalibration_20260720_0250`；artifact 根为
`$REWRITE_ARTIFACT_ROOT/phase1/phase1_target_recalibration_20260720_0250`。

| session | gain | FX/lowered/execution | winner | contaminated |
| --- | ---: | --- | --- | --- |
| qwen_recal_s01 | 6.813% | 6/4/4 | exec_694a3e9839b6ac43 | false |
| qwen_recal_s02 | 6.933% | 6/4/4 | exec_694a3e9839b6ac43 | false |
| qwen_recal_s03 | 6.979% | 6/4/4 | exec_694a3e9839b6ac43 | false |

三个 session 均为独立 Python 进程、独立 cold cache、source clean、monitor-off、
timing sample_count=0、foreign PID 为空、contaminated round ratio=0。formal
boundary clock 均从 1410 MHz 到 1380 MHz；diagnostic boundary 为
1365-1395 MHz。该变化已记录，未通过挑选 workload 或修改 noise floor 处理。

execution class 映射与 canonical representative：

| execution class | canonical | candidates |
| --- | --- | --- |
| exec_241eedbb848ce50e | p0_baseline_separate_silu | p0, p2 |
| exec_239a10e65d521e49 | p1_separate_manual_silu | p1 |
| exec_694a3e9839b6ac43 | p3_fused_chunk_silu | p3, p4 |
| exec_0bf2a7f792dc1e22 | p5_fused_chunk_manual_silu | p5 |

跨 session 结果：fingerprint 与 mapping 全部稳定；winner reproducibility=1.0；
4 strict / 2 tie / 0 ambiguous；strict order reproducibility=1.0；p3/p4 collapse；
p3/p5 relative delta 为 0.215%-0.262%，在锁定 2% floor 下仍为 tie。目标 median
baseline-to-best gain 为 6.933%，最小为 6.813%；七项 Phase 1 gate 全部为 true。

## 与源服务器 Phase 1 的差异

- 硬件仍为 A100 80GB，PyTorch/CUDA 仍为 2.10.0+cu129/12.9；driver 从
  575.57.08 变为 560.35.05，fingerprint schema 从历史 v2 变为 v3。
- 6 FX、4 lowered、4 execution 以及所有 execution hash/class mapping 与历史一致。
- 源端 median gain 为 6.092%，目标端为 6.933%，增加约 0.841 个百分点；两端
  winner 和 strict/tie 结构不变。
- 目标端四个 class 的 median latency 比源端快约 1.60%-2.39%。该差异可能来自
  driver、服务器状态和 boundary clock，不直接并入源端 Phase 1 主统计。
- 因 driver 变化，目标端按指南运行了三个独立 session，而不是用单次 canary
  重估噪声；2% floor 保持不变。

## Phase 2 门禁

目标端跨 session exit decision 为 `pass_phase1`，允许进入 Phase 2。Phase 2
必须从包含本日志、README 环境说明、迁移自检和 compact registry 的干净新 commit
启动。形成该 commit 前 Phase 2 session 列表为空。
