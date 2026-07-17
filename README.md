# Noise-Aware Rewrite Selection for LLM Inference

This repository studies relative selection among semantically equivalent
PyTorch FX rewrites for modern GPU LLM inference. The current research target
is a noise-aware, multi-fidelity policy that reduces lowering, compilation, and
profiling cost while approaching a measured oracle.

The earlier `vertify/day1-day5` work is retained as historical pilot evidence.
It is not part of the formal dataset and must not be used to claim that a GNN
is necessary. The canonical decision document is
[`docs/rewrite_research_plan.md`](docs/rewrite_research_plan.md).

## Current Status

- Phase 0: local engineering baseline established; advisor acceptance remains
  an external exit gate.
- Phase 1: randomized blocked profiling, independent-session output,
  multi-input equivalence checks, environment manifests, and three-level
  fingerprints are under active validation.
- Model training is intentionally blocked until measurement stability,
  lowering retention, and fixed-plan regret pass the research gates.

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
writes a separate session directory and never overwrites historical pilot data.

## Repository Layout

- `configs/`: versioned workloads, rewrites, and profiling protocols.
- `src/rewrite_selector/`: current implementation.
- `scripts/`: experiment entry points and aggregation.
- `tests/`: CPU tests and opt-in GPU integration tests.
- `docs/`: research plan, evidence inventory, and decision log.
- `artifacts/`: generated results, ignored by Git except for the registry.
- `vertify/`: immutable historical pilot.


## License

Released under the [MIT License](LICENSE). Generated profiling artifacts and model weights are not part of the source distribution.
