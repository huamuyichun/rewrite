from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def _trace(plan: dict[str, Any]) -> list[str]:
    return [str(item) for item in plan.get("rewrite_trace", [])]


def build_execution_classes(
    plans: list[dict[str, Any]],
    audits: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    plans_by_id = {str(plan["candidate_id"]): plan for plan in plans}
    grouped: dict[str, list[str]] = {}
    for candidate_id, audit in sorted(audits.items()):
        if audit.get("status") != "ok":
            continue
        execution_hash = audit.get("lowered", {}).get("execution_sha256")
        key = execution_hash or f"missing:{candidate_id}"
        grouped.setdefault(key, []).append(candidate_id)

    classes: list[dict[str, Any]] = []
    for execution_hash, candidate_ids in sorted(grouped.items()):
        ordered = sorted(
            candidate_ids,
            key=lambda candidate_id: (
                len(_trace(plans_by_id[candidate_id])),
                _trace(plans_by_id[candidate_id]),
                candidate_id,
            ),
        )
        representative = ordered[0]
        stable_suffix = (
            execution_hash[:16]
            if not execution_hash.startswith("missing:")
            else representative
        )
        classes.append(
            {
                "execution_class_id": f"exec_{stable_suffix}",
                "execution_sha256": (
                    execution_hash
                    if not execution_hash.startswith("missing:")
                    else None
                ),
                "canonical_candidate_id": representative,
                "candidate_ids": ordered,
                "candidate_rewrite_traces": {
                    candidate_id: _trace(plans_by_id[candidate_id])
                    for candidate_id in ordered
                },
                "num_candidates": len(ordered),
                "fingerprint_status": (
                    "ok"
                    if not execution_hash.startswith("missing:")
                    else "missing"
                ),
            }
        )
    return classes


def representative_callables(
    execution_classes: list[dict[str, Any]],
    callables: dict[str, Callable[[torch.Tensor], torch.Tensor]],
) -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    return {
        execution_class["execution_class_id"]: callables[
            execution_class["canonical_candidate_id"]
        ]
        for execution_class in execution_classes
    }


def class_by_candidate(
    execution_classes: list[dict[str, Any]],
) -> dict[str, str]:
    return {
        candidate_id: execution_class["execution_class_id"]
        for execution_class in execution_classes
        for candidate_id in execution_class["candidate_ids"]
    }


def analyze_fingerprint_consistency(
    execution_classes: list[dict[str, Any]],
    candidate_summary: dict[str, dict[str, float | int]],
    noise_floor_relative: float,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for execution_class in execution_classes:
        measured = [
            candidate_id
            for candidate_id in execution_class["candidate_ids"]
            if candidate_id in candidate_summary
        ]
        values = {
            candidate_id: float(candidate_summary[candidate_id]["p50_ms"])
            for candidate_id in measured
        }
        if len(values) < 2:
            relative_spread = 0.0
        else:
            relative_spread = max(values.values()) / min(values.values()) - 1.0
        diagnostics.append(
            {
                "execution_class_id": execution_class["execution_class_id"],
                "candidate_p50_ms": values,
                "relative_spread": relative_spread,
                "noise_floor_relative": noise_floor_relative,
                "status": (
                    "fingerprint_inconsistency"
                    if len(values) >= 2
                    and relative_spread > noise_floor_relative
                    else "consistent"
                ),
            }
        )
    return diagnostics


def class_summary_from_profile(
    execution_classes: list[dict[str, Any]],
    class_profile: dict[str, dict[str, float | int]],
) -> list[dict[str, Any]]:
    by_id = {
        execution_class["execution_class_id"]: execution_class
        for execution_class in execution_classes
    }
    rows: list[dict[str, Any]] = []
    for execution_class_id, summary in class_profile.items():
        execution_class = by_id[execution_class_id]
        rows.append(
            {
                "execution_class_id": execution_class_id,
                "canonical_candidate_id": execution_class[
                    "canonical_candidate_id"
                ],
                "candidate_ids": execution_class["candidate_ids"],
                **summary,
            }
        )
    return sorted(rows, key=lambda row: float(row["p50_ms"]))
