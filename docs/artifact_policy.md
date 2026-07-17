# Artifact Policy

The Git repository contains source, configs, schemas, tests, documentation, and
compact versioned reports. Raw samples, generated Inductor/Triton code, compiler
caches, model weights, and interrupted session binaries remain local.

Each successful or aborted experiment records:

- `run_id` and `session_id`;
- status (`ok`, `aborted`, `failed`, or `contaminated`);
- source commit and dirty state;
- resolved config hash;
- environment manifest path;
- raw artifact path;
- reason for exclusion when status is not `ok`.

Compact JSON/Markdown reports may be promoted from local artifacts into
`docs/reports/` after schema and lineage checks. Promotion copies only
aggregated evidence, never compiler caches or raw timing samples.
