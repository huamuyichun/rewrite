# Rewrite 项目服务器迁移执行手册

- 文档状态：目标服务器迁移的规范入口
- 适用对象：在全新目标服务器上接手仓库的 Codex
- 唯一研究总计划：`docs/rewrite_research_plan.md`
- 迁移原则：先恢复可复现性，再恢复 GPU 结论，最后进入 Phase 2

## 0. 不可改变的研究边界

1. 不恢复已删除的 GNN、`vertify`、`rewrite_miniexp` 或旧 pilot 路线。
2. Phase 2 决策前不训练 GNN、MLP、tree ranker、DeepSets 或其他 selector。
3. 不修改锁定的 2% noise/tie floor 和 0.5% monitor self-effect gate。
4. 正式 timing 内 `monitor_mode=off`，只保留 NVML boundary snapshots。
5. 每个正式 session 使用独立 Python 进程和独立 cold Inductor cache。
6. GPU 实验只使用一张无 foreign compute PID 的明确空闲卡。
7. 不恢复或汇总 `qwen_s03`。
8. 不把迁移后的单次测量直接并入源服务器 Phase 1 主统计。

本次迁移把 lowered/generated-code fingerprint schema 升级为
`inductor-ir-v3`。v3 会归一化 workspace、artifact、Python prefix、Inductor、
Triton、临时目录和缓存目录。旧 Phase 1 报告仍是 v2 历史证据；跨 schema
不能只比较 hash 字符串，必须先在目标服务器运行 recalibration canary。

## 1. 目标目录模型

不要在代码或文档中写死目标服务器用户名和绝对路径。只选择一个本机高速盘，
然后通过环境变量描述目录：

- `REWRITE_ROOT`：Git clone 后的仓库根目录。
- `LOCAL_DATA_ROOT`：目标服务器本机高速数据根目录。
- `REWRITE_ARTIFACT_ROOT`：raw sessions、Inductor trace 和 session cache。
- `REWRITE_REGISTRY_PATH`：紧凑实验 registry；默认保留在 Git 仓库中。
- `QWEN_MODEL_DIR`：Qwen2.5-7B-Instruct 权重目录。
- `TMPDIR/XDG_CACHE_HOME/HF_HOME/TORCH_HOME/TRITON_CACHE_DIR`：本地高速缓存。

推荐拓扑：

```text
<fast-work-root>/rewrite/                 # Git 仓库
<fast-data-root>/rewrite-data/
  artifacts/                             # raw profiling artifacts
  cache/                                 # pip/HF/Torch/Triton cache
  models/Qwen2.5-7B-Instruct/            # 模型权重
  tmp/
  envs/rewrite/                          # Conda prefix，可选
```

仓库内的 `.env.example` 是可 source 的模板。`.env.migration` 和
`.local-data/` 已被 Git 忽略。

## 2. 阶段 A：目标服务器与存储审计

先检查，不安装、不运行 GPU workload：

```bash
uname -a
uname -m
df -h
nvidia-smi
git --version
ssh -T git@github.com
```

选择本机高速盘，不要把 artifacts、模型、Conda、Inductor 或 Hugging Face
缓存放在慢速共享 home。记录：

- 主机名与操作系统。
- GPU 型号、显存、driver。
- 本机高速盘路径和可用空间。
- 是否存在 Conda/Mamba。
- GitHub SSH 是否可用。

停止条件：

- 没有足够本地空间。
- GPU driver 无法支持计划安装的 PyTorch CUDA runtime。
- GitHub SSH 或目标仓库不可访问。

## 3. 阶段 B：clone 与路径配置

在选定的高速工作根目录执行：

```bash
git clone git@github.com:huamuyichun/rewrite.git
cd rewrite
git checkout main
git pull --ff-only origin main
git status --short --branch
git log -3 --oneline --decorate
```

然后创建目标服务器本地配置：

```bash
cp .env.example .env.migration
```

编辑 `.env.migration`，至少把 `LOCAL_DATA_ROOT` 指向本机高速盘。之后：

```bash
source .env.migration
mkdir -p "$REWRITE_ARTIFACT_ROOT" "$QWEN_MODEL_DIR"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME"
mkdir -p "$TORCH_HOME" "$TRITON_CACHE_DIR"
test "$(pwd -P)" = "$REWRITE_ROOT"
```

每个新 shell/tmux session 都必须先进入仓库并 `source .env.migration`。
不要提交 `.env.migration`。

通过条件：

- `REWRITE_ROOT` 等于当前 clone 根目录。
- artifact、tmp 和 cache 的现有父目录可写。
- `git status` 干净。

## 4. 阶段 C：重建 Python/PyTorch 环境

参考源环境：

- Python 3.12.13。
- PyTorch 2.10.0+cu129。
- CUDA runtime 12.9。
- 源服务器 driver 575.57.08。
- 源服务器正式主卡为 A100 80GB PCIe。

优先使用目标服务器已有 Conda/Mamba。示例：

```bash
conda create -p "$LOCAL_DATA_ROOT/envs/rewrite" python=3.12 pip -y
conda activate "$LOCAL_DATA_ROOT/envs/rewrite"
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu129
python -m pip install -e '.[dev]'
```

如果目标服务器没有 Conda，可把 Miniforge 安装到 `LOCAL_DATA_ROOT/tools`，
不要默认安装到共享 home。安装器 URL 必须根据 `uname -m` 选择；下载后先
核对官方 checksum，再以 batch 模式安装。

验证：

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda runtime", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
PY
python -m pip check
```

如果无法安装完全相同的 PyTorch/CUDA 组合：

1. 不要静默换版本后直接进入 Phase 2。
2. 在迁移日志中记录目标版本、driver 和原因。
3. 仍需跑完整 Phase 1 recalibration。
4. 所有 v2/v3 execution class 映射都视为待重新验证。

## 5. 阶段 D：准备 Qwen 权重

当前 MLP/RMSNorm runner 只使用 Qwen 的维度和随机初始化参数，Phase 1
recalibration 与 Phase 2 microbenchmark 不读取真实 Qwen 权重。因此权重缺失
不应阻塞这些实验，但应在后续 Chitu/production-like 验证前准备好。

推荐用 Hugging Face Hub 下载到 `QWEN_MODEL_DIR`：

```bash
python -m pip install 'huggingface_hub[cli]'
python - <<'PY'
from huggingface_hub import model_info
print(model_info("Qwen/Qwen2.5-7B-Instruct").sha)
PY
```

记录输出的 immutable revision，然后执行：

```bash
hf download Qwen/Qwen2.5-7B-Instruct \
  --revision <HF_COMMIT_SHA> \
  --local-dir "$QWEN_MODEL_DIR"
```

不要把 token 写进仓库，不要提交权重。验证：

```bash
python scripts/check_migration.py --require-qwen
```

检查项包括：

- `config.json` 存在。
- `hidden_size=3584`。
- `intermediate_size=18944`。
- 至少一个 safetensors/bin 权重文件存在。

## 6. 阶段 E：决定是否同步 Phase 1 raw artifacts

Git 已包含 compact registry 和 Phase 1 正式报告。下面两类任务不要求复制 raw
artifacts：

- 运行目标服务器 recalibration canary。
- 从零开始 Phase 2 discovery。

只有以下需求才建议同步：

- 审计 v2 的原始 rounds/samples。
- 比较目标服务器 v3 与源服务器 v2 lowering/codegen。
- 重跑 Phase 1 聚合或核查环境 manifest。

最低建议同步集：

```text
artifacts/phase1/phase1_qwen_decode_monitor_off_20260717/qwen_s06
artifacts/phase1/phase1_qwen_decode_monitor_off_20260717/qwen_s07
artifacts/phase1/phase1_qwen_decode_monitor_off_20260717/qwen_s08
artifacts/phase1/phase1_monitor_self_effect_clean_long_20260717
artifacts/phase1/phase1_monitor_self_effect_clean_2s_5cycle_20260717
artifacts/phase1/phase1_monitor_self_effect_clean_3s_5cycle_20260717
```

从目标服务器执行 rsync，变量由操作者填写：

```bash
export SOURCE_HOST=<source-host>
export SOURCE_ARTIFACT_ROOT=<source-artifact-root>

rsync -a --info=progress2 \
  "$SOURCE_HOST:$SOURCE_ARTIFACT_ROOT/phase1/phase1_qwen_decode_monitor_off_20260717" \
  "$REWRITE_ARTIFACT_ROOT/phase1/"

rsync -a --info=progress2 \
  "$SOURCE_HOST:$SOURCE_ARTIFACT_ROOT/phase1/phase1_monitor_self_effect_clean_long_20260717" \
  "$SOURCE_HOST:$SOURCE_ARTIFACT_ROOT/phase1/phase1_monitor_self_effect_clean_2s_5cycle_20260717" \
  "$SOURCE_HOST:$SOURCE_ARTIFACT_ROOT/phase1/phase1_monitor_self_effect_clean_3s_5cycle_20260717" \
  "$REWRITE_ARTIFACT_ROOT/phase1/"
```

禁止使用 `rsync --delete`。不要覆盖 Git clone 带来的
`artifacts/registry.jsonl`。同步后执行：

```bash
python scripts/check_migration.py --require-phase1-artifacts
```

如果需要逐文件传输校验，在源端和目标端分别从各自 artifact root 执行：

```bash
(
  cd "$REWRITE_ARTIFACT_ROOT"
  find phase1/phase1_qwen_decode_monitor_off_20260717 \
    -type f -print0 | sort -z | xargs -0 sha256sum
) > "$REWRITE_ARTIFACT_ROOT/phase1_qwen_sha256.txt"
```

源端使用自己的 artifact root 运行同一命令。比较两份清单；清单只保存在
artifact root，不提交大型 raw 文件。

## 7. 阶段 F：CPU 验证与 dry run

先确保 GPU 对当前进程不可见：

```bash
export CUDA_VISIBLE_DEVICES=""
source .env.migration
python scripts/check_migration.py
python -m compileall -q src scripts tests
PYTHONPATH=src python -m pytest -q
```

历史接手点是 23 项测试；迁移改造新增测试后，当前目标为 28 passed。以后测试
继续增加时，以 clone commit 的完整测试集全通过为准，不要写死旧数量。

统一枚举 CLI dry run：

```bash
python scripts/enumerate_candidates.py \
  --config configs/rewrites/mlp_bounded_v1.json \
  --output "$TMPDIR/mlp_enumeration.json"

python scripts/enumerate_candidates.py \
  --config configs/rewrites/rmsnorm_bounded_v1.json \
  --output "$TMPDIR/rmsnorm_enumeration.json"
```

期望：

- MLP：19 candidates，growth 1/4/10/19。
- RMSNorm：8 candidates，growth 1/4/8。
- CPU runner 的两族端到端 dry run 已包含在 pytest 中。
- `git diff --check` 通过。

如果输出文件已经存在，换新文件名；不要覆盖旧证据。

## 8. 阶段 G：GPU 门禁

重新打开一个 shell，source 环境，只暴露一张候选卡：

```bash
source .env.migration
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu,clocks.sm \
  --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits

export CUDA_VISIBLE_DEVICES=<one-physical-gpu-index>
python scripts/check_migration.py --require-gpu --require-clean-git
```

通过条件：

- `CUDA_VISIBLE_DEVICES` 只有一个数字物理 index token。
- `torch.cuda.is_available()` 为 true。
- NVML snapshot 中 `foreign_processes=[]`。
- 自检 snapshot 的 GPU utilization 不高于 5%。
- 目标卡显存状态符合空闲预期。
- Git source state 干净。

任何条件失败都不运行 smoke，不排队，不抢占，不碰其他用户进程。

## 9. 阶段 H：GPU smoke

smoke 只验证 CUDA、Inductor、NVML、fingerprint v3、registry 和写盘链路，
不生成正式性能结论：

```bash
export RUN_ID="migration_gpu_smoke_$(date +%Y%m%d_%H%M%S)"

python scripts/run_phase1_audit.py \
  --rewrites configs/rewrites/mlp_control_v1.json \
  --workloads configs/workloads/phase1_pilot_v1.json \
  --protocol configs/profiling/migration_gpu_smoke_v1.json \
  --group-id pilot_b0_seq128_h1024_i4096_fp16 \
  --run-id "$RUN_ID" \
  --session-id smoke_s01 \
  --output-root "$REWRITE_ARTIFACT_ROOT/migration" \
  --registry "$REWRITE_ARTIFACT_ROOT/migration/registry.jsonl"
```

检查：

```bash
jq . "$REWRITE_ARTIFACT_ROOT/migration/$RUN_ID/smoke_s01/status.json"
jq . "$REWRITE_ARTIFACT_ROOT/migration/$RUN_ID/smoke_s01/session_summary.json"
```

通过条件：

- status 为 `ok`。
- `source_dirty_at_run=false`。
- 6 个 candidates 全部通过 equivalence。
- fingerprint schema 为 `inductor-ir-v3`，lowered/execution hash 非空。
- `contaminated=false`。
- monitor mode 为 off，timing 内 sample_count=0。
- 前后 boundary snapshots 存在且无 foreign PID。

smoke 结果不进入 Phase 1/2 latency 汇总。

## 10. 阶段 I：目标服务器 Phase 1 recalibration canary

Qwen 权重不是此命令的输入；这里复现的是 Qwen2.5-7B MLP 维度。

```bash
export RUN_ID="phase1_target_recalibration_$(date +%Y%m%d_%H%M%S)"

python scripts/run_phase1_audit.py \
  --rewrites configs/rewrites/mlp_control_v1.json \
  --workloads configs/workloads/phase1_pilot_v1.json \
  --protocol configs/profiling/phase1_recalibration_v1.json \
  --group-id qwen2p5_7b_decode_bs1_t1_bf16 \
  --run-id "$RUN_ID" \
  --session-id qwen_recal_s01 \
  --output-root "$REWRITE_ARTIFACT_ROOT/phase1" \
  --registry "$REWRITE_REGISTRY_PATH"
```

必须审计：

- 6 个 control candidates 的 eager/compiled equivalence。
- FX/lowered/execution unique count。
- execution class 映射与 canonical representative。
- baseline-to-best gain、spread 和 same-class diagnostic。
- monitor mode、boundary clocks、foreign PID 和 contaminated rounds。
- source commit/dirty state、driver、GPU UUID、PyTorch/CUDA 和 cache policy。

在相同 A100/PyTorch/CUDA 组合上，历史参考是 6 FX、4 lowered、4 execution，
p0/p2 collapse、p3/p4 collapse，baseline-to-best 约 6%。这些是诊断参考，
不是强行通过标准；目标结果不同必须先解释 compiler、GPU、clock、fingerprint
schema 或噪声差异，不能挑选 workload 修正结论。

单个 canary 只能证明迁移链路可用，不能重新估计跨 session noise。如果目标硬件或
软件版本变化，至少运行 3 个独立 cold-cache、monitor-off session 后，才能形成
新的目标服务器测量基线；不得根据单次结果修改 2% floor。

canary 审计完成后，新增一份中文目标服务器迁移日志，只暂存 compact
`artifacts/registry.jsonl`、日志和必要的小型报告，运行测试后形成中文 commit
并 push。不要提交 raw session、Inductor/Triton cache 或模型权重。Phase 2 必须
基于这个干净的新 commit 启动。

## 11. 阶段 J：进入 Phase 2

只有 CPU、GPU smoke 和 Phase 1 recalibration 均通过，才启动第一组 RMSNorm
discovery：

```bash
export RUN_ID="phase2_rmsnorm_discovery_$(date +%Y%m%d_%H%M%S)"

python scripts/run_phase1_audit.py \
  --rewrites configs/rewrites/rmsnorm_bounded_v1.json \
  --workloads configs/workloads/phase2_rmsnorm_discovery_v1.json \
  --protocol configs/profiling/phase2_discovery_decode_v1.json \
  --group-id phase2_rmsnorm_norm_only_decode_bs1_t1_bf16 \
  --run-id "$RUN_ID" \
  --session-id rms_d01 \
  --output-root "$REWRITE_ARTIFACT_ROOT/phase2" \
  --registry "$REWRITE_REGISTRY_PATH"
```

先审计单组的：

- 8 个 candidates 是否全部数值合法。
- FX/lowered/execution retention 与 collapse classes。
- execution-class spread 是否超过噪声。
- winner/tie 是否有意义。
- monitor boundary 和污染状态。
- fingerprint v3 是否稳定。

只有该组干净，才扩到其余 RMSNorm groups；先看结果再决定是否运行 9 个 MLP
control groups，不机械跑完 17 组。仍不允许训练模型。

## 12. 失败处理

| 失败 | 动作 |
| --- | --- |
| Python/PyTorch 版本不一致 | 停止 GPU 实验，先记录和决定是否接受新 compiler domain |
| CUDA unavailable | 检查 driver、wheel 和 CUDA_VISIBLE_DEVICES，不改研究配置 |
| Qwen 权重缺失 | microbenchmark 可继续；真实模型/Chitu 验证暂停 |
| raw artifacts 无法同步 | 保留 compact reports，重新跑目标 canary，不伪造历史 raw |
| CPU 测试失败 | 修复后重新从完整测试开始 |
| fingerprint v3 不稳定 | 停止 profiling，先修归一化与测试 |
| GPU 有 foreign PID | 不运行、不排队，等待其他卡或其他时段 |
| smoke contaminated | 标记并排除；重新选择空闲卡和新 session_id |
| canary 映射变化 | 不进入 Phase 2，先做 compiler/hardware/fingerprint 审计 |
| registry 已有同名 session | 换新的 run_id/session_id，不覆盖目录 |

## 13. 目标服务器 Codex 的必读顺序

1. 本文件 `docs/SERVER_MIGRATION_GUIDE.md`。
2. `docs/rewrite_research_plan.md`。
3. `docs/NEXT_CODEX_PROMPT.md`。
4. `README.md`、`environment/README.md`、`docs/artifact_policy.md`。
5. `docs/reports/phase1/` 下全部 Markdown/JSON。
6. 当前代码、测试和配置。

目标 Codex 完成迁移后必须用中文报告：

- clone commit 与 origin/main 同步状态。
- 目标目录变量，但不包含 secret/token。
- Python/PyTorch/CUDA/driver/GPU UUID。
- Qwen revision 与检查结果。
- 是否同步 raw Phase 1 artifacts、同步路径和 checksum 结果。
- CPU 测试、compileall、enumeration dry run。
- GPU smoke session 与 artifact 路径。
- recalibration session、污染状态、fingerprint 与 execution class 映射。
- 与历史 Phase 1 的差异。
- 是否允许进入 Phase 2。
- Phase 2 已启动的完整 session 列表。
