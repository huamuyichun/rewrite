# Generated Artifacts

Experiment outputs are ignored by Git. Each Phase 1 invocation writes to
`artifacts/phase1/<run_id>/<session_id>/` and appends a compact entry to
`artifacts/registry.jsonl` only after successful completion.
