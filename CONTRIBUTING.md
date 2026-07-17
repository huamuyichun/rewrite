# Contributing

This research repository keeps source, compact reports, and local raw
artifacts separate.

Before submitting a change:

1. Run `PYTHONPATH=src python -m pytest -q`.
2. Add tests for rewrite preconditions, equivalence, fingerprints, schemas, or
   metrics as applicable.
3. Do not commit model weights, compiler caches, or raw profiling runs.
4. Do not treat repeated measurements, random weights, or lowering-collapsed
   candidates as independent training samples.
5. Record configs, environment fingerprints, commit hashes, and artifact paths
   in the experiment registry.

Research claims must follow `docs/rewrite_research_plan.md`.
