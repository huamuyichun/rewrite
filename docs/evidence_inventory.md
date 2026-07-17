# Evidence Inventory

## Historical Pilot

The `vertify/day1-day5` pipeline is historical evidence only. Its formal
Inductor result contains three workload groups and six hand-authored MLP plans.
All six FX signatures are distinct, but no lowered or execution fingerprint was
captured. Fixed `p3_fused_chunk_silu` has zero median regret and about 2.38%
maximum regret on those groups. This is direct counter-evidence against moving
to a learned selector on the current space.

| Evidence | Supports | Does not support |
| --- | --- | --- |
| 6 executable FX candidates | candidate construction works | 6 distinct executions |
| single-input allclose | basic numerical sanity | semantic equivalence over input domain |
| 3-group Inductor timing | measurable pilot spread | generalization or model necessity |
| p3 near-oracle fixed rule | strong simple baseline | learned-selector value |

## Current Required Evidence

Phase 1 must produce independent-session order reproducibility, bootstrap
intervals narrower than the target effect, automatic contamination flags, and
stable high-level/lowered/execution fingerprints. Phase 2 is blocked until
these conditions are evaluated from real results.

## Claim Guardrail

The project must not claim first use of GNNs, learned cost models, pairwise
ranking, gate/up merging, or fused SiLU-and-multiply for graph rewrite
selection. The active differentiation is modern open GPU/LLM/Inductor data,
noise-aware relative decisions, and selective multi-fidelity measurement.
