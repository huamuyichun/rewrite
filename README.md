# 面向 LLM 推理的噪声感知 Rewrite 选择

本仓库研究现代 GPU LLM 推理中语义等价 PyTorch FX rewrite 的 lowering
塌缩、execution class、真实性能差异和噪声感知选择。唯一研究总计划是
[`docs/rewrite_research_plan.md`](docs/rewrite_research_plan.md)。

在全新服务器上接手时，必须首先完整执行
[`docs/SERVER_MIGRATION_GUIDE.md`](docs/SERVER_MIGRATION_GUIDE.md)，不要沿用
源服务器绝对路径。

## 当前状态

- Phase 0：研究边界与可复现工程基线完成。
- Phase 1：正式关闭；三个 Qwen monitor-off session 复现四个 execution
  classes 和约 6% baseline-to-best gain。
- Phase 2：MLP/RMSNorm family adapter、17 组 discovery 配置和 CPU dry run
  已完成；尚未训练任何模型。
- 跨服务器 fingerprint schema 已升级为 `inductor-ir-v3`，目标服务器需先
  重新运行 Phase 1 recalibration canary。

旧 GNN、`vertify`、`rewrite_miniexp` 和手工 pilot 已删除，不得恢复。

## 可移植环境

仓库代码通过脚本位置自定位，不依赖 clone 的绝对路径。目标服务器从模板创建
本地环境变量：

```bash
cp .env.example .env.migration
source .env.migration
```

推荐 Python 3.12、PyTorch 2.10.0+cu129、CUDA runtime 12.9。详细版本和迁移
门禁见 [`environment/README.md`](environment/README.md) 与迁移手册。

## 快速 CPU 检查

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m compileall -q src scripts tests
python scripts/check_migration.py
python scripts/run_phase1_audit.py --help
```

GPU 实验前必须只暴露一张明确空闲卡，并同时检查 GPU 状态和 compute PID。
每次 invocation 写入独立 session 目录，永不覆盖已有 artifact。

## 仓库结构

- `configs/`：版本化 workload、rewrite 和 profiling protocol。
- `src/rewrite_selector/`：当前实现。
- `scripts/`：实验、分析和迁移自检入口。
- `tests/`：CPU 回归和可选 GPU integration tests。
- `docs/`：唯一研究计划、迁移手册、报告与接力文档。
- `environment/`：参考环境说明。
- `artifacts/`：只跟踪 compact registry；raw artifacts 默认不入 Git。

## License

代码采用 [MIT License](LICENSE)。生成的 profiling artifacts、模型权重和
compiler cache 不属于源码分发内容。
