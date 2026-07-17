#!/usr/bin/env python3
"""Validate the Day 3 candidate rewrite space definition.

This is a schema/consistency check only. It does not instantiate executable
graphs and does not profile latency.
"""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_FIELDS = {
    "candidate_id",
    "is_baseline",
    "rewrite_family",
    "plan_desc",
    "gate_up_projection",
    "gate_up_split",
    "activation",
    "multiply",
    "expected_fx_pattern",
    "semantic_condition",
    "structural_signature",
}


def main() -> None:
    root = Path(__file__).resolve().parent
    plans = json.loads((root / "candidate_plans.json").read_text(encoding="utf-8"))

    errors: list[str] = []
    if not isinstance(plans, list):
        errors.append("candidate_plans.json must contain a list")
        plans = []

    candidate_ids = [p.get("candidate_id") for p in plans if isinstance(p, dict)]
    signatures = [p.get("structural_signature") for p in plans if isinstance(p, dict)]
    families = sorted({p.get("rewrite_family") for p in plans if isinstance(p, dict)})
    baseline_ids = [p.get("candidate_id") for p in plans if isinstance(p, dict) and p.get("is_baseline")]

    for idx, plan in enumerate(plans):
        if not isinstance(plan, dict):
            errors.append(f"plan at index {idx} is not an object")
            continue
        missing = sorted(REQUIRED_FIELDS - set(plan))
        if missing:
            errors.append(f"{plan.get('candidate_id', idx)} missing fields: {missing}")

    if len(plans) < 4:
        errors.append("candidate count must be at least 4")
    if len(plans) > 8:
        errors.append("candidate count should stay within the Day 3 target range 4-8")
    if len(set(candidate_ids)) != len(candidate_ids):
        errors.append("candidate_id values must be unique")
    if len(set(signatures)) != len(signatures):
        errors.append("structural_signature values must be unique")
    if len(baseline_ids) != 1:
        errors.append("there must be exactly one baseline candidate")
    if len(families) != 1:
        errors.append("Day 3 should use exactly one rewrite family")

    result = {
        "status": "ok" if not errors else "failed",
        "num_candidates": len(plans),
        "num_baselines": len(baseline_ids),
        "baseline_ids": baseline_ids,
        "rewrite_families": families,
        "num_unique_candidate_ids": len(set(candidate_ids)),
        "num_unique_structural_signatures": len(set(signatures)),
        "errors": errors,
    }

    (root / "validation_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
