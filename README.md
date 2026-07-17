# Noise-Aware Rewrite Selection for LLM Inference

This repository studies relative selection among semantically equivalent
PyTorch FX rewrites for modern GPU LLM inference. The current research target
is a noise-aware, multi-fidelity policy that reduces lowering, compilation, and
profiling cost while approaching a measured oracle.

The obsolete hand-written GNN-era pilot tree was removed after its useful
control plans were migrated into versioned configs. It is not part of the
formal dataset. The canonical decision document is
[`docs/rewrite_research_plan.md`](docs/rewrite_research_plan.md).

## Current Status

- Phase 0: research scope and reproducible engineering baseline established.
- Phase 1: complete. Independent Qwen sessions reproduce four execution
  classes and about 6% baseline-to-best gain; periodic monitoring was found to
  perturb timing and is disabled inside formal timing windows.
- Phase 2: bounded candidate-family discovery is next. No model was trained.

## Environment

The reference environment is
`/pub/data/hjwz/miniconda3/envs/rewrite_miniexp` with Python 3.12,
PyTorch 2.10.0+cu129, CUDA 12.9, and an NVIDIA A100. Core versions and the
reproduction command are recorded in [`environment/README.md`](environment/README.md).

## Quick Checks

```bash
PYTHONPATH=src /pub/data/hjwz/miniconda3/envs/rewrite_miniexp/bin/python -m pytest -q
/pub/data/hjwz/miniconda3/envs/rewrite_miniexp/bin/python scripts/run_phase1_audit.py --help
```

GPU experiments must use one explicitly selected idle GPU. Every invocation
writes a separate session directory and never overwrites existing artifacts.

## Repository Layout

- `configs/`: versioned workloads, rewrites, and profiling protocols.
- `src/rewrite_selector/`: current implementation.
- `scripts/`: experiment entry points and aggregation.
- `tests/`: CPU tests and opt-in GPU integration tests.
- `docs/`: canonical research plan, reports, and handoff material.
- `artifacts/`: generated results, ignored by Git except for the registry.


## License

Released under the [MIT License](LICENSE). Generated profiling artifacts and model weights are not part of the source distribution.
