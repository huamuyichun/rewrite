# 生成的 Artifacts

raw 实验输出默认不进入 Git。runner 默认写入
`${REWRITE_ARTIFACT_ROOT:-artifacts}/phase1/<run_id>/<session_id>/`，也可通过
`--output-root` 显式指定外部高速盘目录。

紧凑 registry 默认是仓库内的 `artifacts/registry.jsonl`，可通过
`REWRITE_REGISTRY_PATH` 或 `--registry` 改写。无论 raw artifact 位于仓库内
还是外部目录，registry 都记录可解析路径；session 目录永不覆盖。
