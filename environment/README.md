# 参考环境与可移植路径

源服务器在 2026-07-16/17 形成 Phase 1 证据时的环境：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12.13 (conda-forge) |
| PyTorch | 2.10.0+cu129 |
| CUDA runtime | 12.9 |
| NVIDIA driver | 575.57.08 |
| 正式主 GPU | NVIDIA A100 80GB PCIe |
| fingerprint | 历史 v2；迁移后代码为 `inductor-ir-v3` |

Conda prefix 是服务器本地状态，不属于可迁移接口。新服务器应创建独立环境，并
通过 `.env.example` 设置 `REWRITE_ROOT`、`REWRITE_ARTIFACT_ROOT`、
`REWRITE_REGISTRY_PATH`、`QWEN_MODEL_DIR` 和缓存路径。

```bash
source .env.migration
PYTHONPATH=src python scripts/run_phase1_audit.py --help
python scripts/check_migration.py
```

每个 session 的 `environment.json` 会重新记录实际 Python、PyTorch、CUDA、
driver、GPU、cache 和路径配置。表格只是历史参考，不能替代 live manifest。
完整重建与验证流程见 `docs/SERVER_MIGRATION_GUIDE.md`。
