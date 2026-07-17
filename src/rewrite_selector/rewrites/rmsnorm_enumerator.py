from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import torch

from rewrite_selector.ir.rmsnorm import (
    RMSNormWorkload,
    instantiate_rmsnorm_candidate,
    make_rmsnorm_baseline,
    make_rmsnorm_input,
)
from rewrite_selector.lowering.fingerprint import high_level_fingerprint


@dataclass(frozen=True, order=True)
class RMSNormState:
    implementation: str = "native"
    flatten: bool = False
    scale_association: str = "left"

    def stable_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


RULES = [
    {
        "rule_id": "rmsnorm.decompose_square_mul",
        "hypothesis": "Explicit multiply/mean/rsqrt exposes pointwise fusion and "
        "intermediate materialization decisions.",
    },
    {
        "rule_id": "rmsnorm.decompose_square_pow",
        "hypothesis": "pow versus multiply may change generated pointwise code.",
    },
    {
        "rule_id": "rmsnorm.flatten_hidden_rows",
        "hypothesis": "Flatten/restore changes view and stride propagation at "
        "the norm boundary.",
    },
    {
        "rule_id": "rmsnorm.reassociate_scale",
        "hypothesis": "Equivalent scale association changes the pointwise DAG "
        "and potential fusion grouping.",
    },
]


def _apply(
    state: RMSNormState,
    rule_id: str,
) -> RMSNormState | None:
    if rule_id == "rmsnorm.decompose_square_mul":
        if state.implementation != "native":
            return None
        return RMSNormState(
            "square_mul",
            state.flatten,
            state.scale_association,
        )
    if rule_id == "rmsnorm.decompose_square_pow":
        if state.implementation != "native":
            return None
        return RMSNormState(
            "square_pow",
            state.flatten,
            state.scale_association,
        )
    if rule_id == "rmsnorm.flatten_hidden_rows":
        if state.flatten:
            return None
        return RMSNormState(
            state.implementation,
            True,
            state.scale_association,
        )
    if rule_id == "rmsnorm.reassociate_scale":
        if state.implementation == "native":
            return None
        if state.scale_association == "right":
            return None
        return RMSNormState(
            state.implementation,
            state.flatten,
            "right",
        )
    raise ValueError(rule_id)


def enumerate_rmsnorm_candidates(
    max_depth: int = 2,
    max_candidates: int = 16,
) -> dict[str, Any]:
    initial = RMSNormState()
    records: dict[RMSNormState, dict[str, Any]] = {
        initial: {"min_depth": 0, "traces": [[]]}
    }
    queue = deque([(initial, [])])
    expanded: set[RMSNormState] = set()
    edges = []
    while queue:
        state, trace = queue.popleft()
        if state in expanded:
            continue
        expanded.add(state)
        if len(trace) >= max_depth:
            continue
        for rule in sorted(RULES, key=lambda item: item["rule_id"]):
            target = _apply(state, rule["rule_id"])
            if target is None:
                continue
            next_trace = [*trace, rule["rule_id"]]
            edges.append(
                {
                    "source_state_hash": state.stable_hash(),
                    "target_state_hash": target.stable_hash(),
                    "rule_id": rule["rule_id"],
                    "depth": len(next_trace),
                }
            )
            record = records.setdefault(
                target,
                {"min_depth": len(next_trace), "traces": []},
            )
            if next_trace not in record["traces"]:
                record["traces"].append(next_trace)
                record["traces"].sort()
            if target not in expanded:
                queue.append((target, next_trace))

    workload = RMSNormWorkload(
        "canonical",
        "synthetic",
        "prefill",
        1,
        2,
        16,
        "fp32",
        20260717,
        "residual_silu",
    )
    device = torch.device("cpu")
    baseline = make_rmsnorm_baseline(workload, device, torch.float32)
    example = make_rmsnorm_input(
        workload,
        device,
        torch.float32,
        workload.seed,
        "normal",
    )
    groups: dict[str, list[tuple[RMSNormState, dict[str, Any]]]] = {}
    for state, record in sorted(
        records.items(),
        key=lambda item: (
            item[1]["min_depth"],
            item[0].stable_hash(),
        ),
    ):
        plan = {
            **asdict(state),
            "candidate_id": f"state_{state.stable_hash()[:12]}",
        }
        module = instantiate_rmsnorm_candidate(
            plan,
            workload,
            baseline,
            device,
            torch.float32,
        )
        fx_hash = high_level_fingerprint(module, example)["sha256"]
        groups.setdefault(fx_hash, []).append((state, record))

    candidates = []
    for fx_hash, members in sorted(groups.items()):
        members.sort(
            key=lambda item: (
                item[1]["min_depth"],
                item[0].stable_hash(),
            )
        )
        state, record = members[0]
        traces = sorted(
            {
                tuple(trace)
                for _, member_record in members
                for trace in member_record["traces"]
            }
        )
        candidates.append(
            {
                **asdict(state),
                "candidate_id": f"rms_fx_{fx_hash[:12]}",
                "is_baseline": state == initial,
                "rewrite_family": "rmsnorm_residual_boundary",
                "rewrite_trace": list(traces[0]),
                "provenance_traces": [list(trace) for trace in traces],
                "min_rewrite_depth": min(len(trace) for trace in traces),
                "fx_sha256": fx_hash,
            }
        )
    candidates.sort(
        key=lambda item: (
            item["min_rewrite_depth"],
            item["candidate_id"],
        )
    )
    return {
        "schema_version": "bounded-rewrite-enumeration-v1",
        "family_id": "rmsnorm_residual_boundary",
        "max_rewrite_depth": max_depth,
        "max_fx_unique_candidates": max_candidates,
        "num_enumerated_states": len(records),
        "num_fx_unique_before_budget": len(candidates),
        "budget_truncated": len(candidates) > max_candidates,
        "candidate_growth_by_depth": {
            str(depth): sum(
                item["min_rewrite_depth"] <= depth
                for item in candidates
            )
            for depth in range(max_depth + 1)
        },
        "rule_registry": RULES,
        "enumeration_tree": edges,
        "candidates": candidates[:max_candidates],
    }
