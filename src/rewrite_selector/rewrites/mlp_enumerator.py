from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch

from rewrite_selector.ir.mlp import (
    Workload,
    instantiate_candidate,
    make_baseline,
    make_input,
)
from rewrite_selector.lowering.fingerprint import high_level_fingerprint


@dataclass(frozen=True, order=True)
class MLPState:
    gate_up_projection: str = "separate"
    gate_up_split: str = "none"
    packing_order: str = "gate_up"
    activation: str = "f_silu"
    multiply: str = "out_of_place"

    def canonical_json(self) -> str:
        return json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        )

    def stable_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


@dataclass(frozen=True)
class RewriteRule:
    rule_id: str
    hypothesis: str
    apply: Callable[[MLPState], tuple[MLPState | None, str | None]]


def _merge_projection(state: MLPState) -> tuple[MLPState | None, str | None]:
    if state.gate_up_projection != "separate":
        return None, "projection already merged"
    return (
        MLPState(
            gate_up_projection="fused",
            gate_up_split="chunk",
            packing_order="gate_up",
            activation=state.activation,
            multiply=state.multiply,
        ),
        None,
    )


def _pack_up_gate(state: MLPState) -> tuple[MLPState | None, str | None]:
    if state.gate_up_projection != "fused":
        return None, "packing order applies only to merged projection"
    if state.packing_order == "up_gate":
        return None, "up_gate packing already active"
    return (
        MLPState(
            **{
                **asdict(state),
                "packing_order": "up_gate",
            }
        ),
        None,
    )


def _set_split(
    state: MLPState,
    split_mode: str,
) -> tuple[MLPState | None, str | None]:
    if state.gate_up_projection != "fused":
        return None, "split expression requires merged projection"
    if state.gate_up_split == split_mode:
        return None, f"{split_mode} split already active"
    return (
        MLPState(
            **{
                **asdict(state),
                "gate_up_split": split_mode,
            }
        ),
        None,
    )


def _decompose_silu(
    state: MLPState,
) -> tuple[MLPState | None, str | None]:
    if state.activation != "f_silu":
        return None, "SiLU already decomposed"
    return (
        MLPState(
            **{
                **asdict(state),
                "activation": "manual_silu",
            }
        ),
        None,
    )


def _use_inplace_multiply(
    state: MLPState,
) -> tuple[MLPState | None, str | None]:
    if state.multiply != "out_of_place":
        return None, "multiply already in-place"
    return (
        MLPState(
            **{
                **asdict(state),
                "multiply": "inplace",
            }
        ),
        None,
    )


def mlp_rule_registry() -> list[RewriteRule]:
    return sorted(
        [
            RewriteRule(
                "mlp.merge_gate_up",
                "Merging shared-input GEMMs can change library selection, "
                "launch count, and intermediate materialization.",
                _merge_projection,
            ),
            RewriteRule(
                "mlp.pack_up_gate",
                "Legal output-channel packing changes split mapping and may "
                "alter generated indexing/layout code.",
                _pack_up_gate,
            ),
            RewriteRule(
                "mlp.split_chunk",
                "chunk, split, and narrow expose different high-level shape "
                "operations and may alter lowering or fusion boundaries.",
                lambda state: _set_split(state, "chunk"),
            ),
            RewriteRule(
                "mlp.split_narrow",
                "Explicit narrow operations can change view/stride handling "
                "and generated indexing.",
                lambda state: _set_split(state, "narrow"),
            ),
            RewriteRule(
                "mlp.split_split",
                "Explicit split can lower differently from chunk or narrow.",
                lambda state: _set_split(state, "split"),
            ),
            RewriteRule(
                "mlp.silu_decompose",
                "Native SiLU versus sigmoid-multiply decomposition may change "
                "pointwise fusion and code generation.",
                _decompose_silu,
            ),
            RewriteRule(
                "mlp.multiply_inplace",
                "A use-count-safe in-place multiply may reduce materialization "
                "or allocation, subject to alias guards.",
                _use_inplace_multiply,
            ),
        ],
        key=lambda rule: rule.rule_id,
    )


def _state_to_plan(state: MLPState) -> dict[str, Any]:
    return {
        **asdict(state),
        "candidate_id": f"state_{state.stable_hash()[:12]}",
        "is_baseline": state == MLPState(),
        "rewrite_family": "mlp_gate_up_activation_control",
    }


def _enumerate_states(
    max_depth: int,
) -> tuple[
    dict[MLPState, dict[str, Any]],
    list[dict[str, Any]],
    Counter[str],
]:
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    initial = MLPState()
    records: dict[MLPState, dict[str, Any]] = {
        initial: {"min_depth": 0, "traces": [[]]}
    }
    queue: deque[tuple[MLPState, list[str]]] = deque([(initial, [])])
    expanded: set[MLPState] = set()
    edges: list[dict[str, Any]] = []
    invalid = Counter()

    while queue:
        state, trace = queue.popleft()
        if state in expanded:
            continue
        expanded.add(state)
        if len(trace) >= max_depth:
            continue
        for rule in mlp_rule_registry():
            next_state, reason = rule.apply(state)
            if next_state is None:
                invalid[f"{rule.rule_id}: {reason}"] += 1
                continue
            next_trace = [*trace, rule.rule_id]
            edges.append(
                {
                    "source_state_hash": state.stable_hash(),
                    "target_state_hash": next_state.stable_hash(),
                    "rule_id": rule.rule_id,
                    "depth": len(next_trace),
                }
            )
            record = records.setdefault(
                next_state,
                {"min_depth": len(next_trace), "traces": []},
            )
            if next_trace not in record["traces"]:
                record["traces"].append(next_trace)
                record["traces"].sort()
            record["min_depth"] = min(
                int(record["min_depth"]),
                len(next_trace),
            )
            if next_state not in expanded:
                queue.append((next_state, next_trace))
    return records, edges, invalid


def enumerate_mlp_candidates(
    max_depth: int = 3,
    max_candidates: int = 32,
) -> dict[str, Any]:
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    records, edges, invalid = _enumerate_states(max_depth)

    workload = Workload(
        "enumeration_canonical",
        "synthetic",
        "prefill",
        1,
        2,
        16,
        32,
        "fp32",
        20260717,
    )
    device = torch.device("cpu")
    baseline = make_baseline(workload, device, torch.float32)
    example = make_input(workload, device, torch.float32, workload.seed, "normal")

    fingerprint_groups: dict[str, list[dict[str, Any]]] = {}
    invalid_states: list[dict[str, Any]] = []
    for state, record in sorted(
        records.items(),
        key=lambda item: (
            int(item[1]["min_depth"]),
            item[0].canonical_json(),
        ),
    ):
        plan = _state_to_plan(state)
        try:
            module = instantiate_candidate(
                plan,
                workload,
                baseline,
                device,
                torch.float32,
            )
            fingerprint = high_level_fingerprint(module, example)
        except Exception as exc:
            invalid_states.append(
                {
                    "state": asdict(state),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        fingerprint_groups.setdefault(fingerprint["sha256"], []).append(
            {
                "state": state,
                "record": record,
                "fingerprint": fingerprint,
            }
        )

    candidates: list[dict[str, Any]] = []
    fx_collapses: list[dict[str, Any]] = []
    for fx_hash, members in sorted(fingerprint_groups.items()):
        members.sort(
            key=lambda member: (
                int(member["record"]["min_depth"]),
                member["state"].canonical_json(),
            )
        )
        canonical = members[0]
        traces = sorted(
            {
                tuple(trace)
                for member in members
                for trace in member["record"]["traces"]
            }
        )
        candidate = {
            **_state_to_plan(canonical["state"]),
            "candidate_id": f"mlp_fx_{fx_hash[:12]}",
            "fx_sha256": fx_hash,
            "min_rewrite_depth": min(len(trace) for trace in traces),
            "rewrite_trace": list(traces[0]),
            "provenance_traces": [list(trace) for trace in traces],
            "provenance_state_hashes": sorted(
                member["state"].stable_hash() for member in members
            ),
            "num_fx_nodes": canonical["fingerprint"]["num_nodes"],
        }
        candidates.append(candidate)
        if len(members) > 1:
            fx_collapses.append(
                {
                    "fx_sha256": fx_hash,
                    "canonical_candidate_id": candidate["candidate_id"],
                    "state_hashes": candidate["provenance_state_hashes"],
                }
            )

    candidates.sort(
        key=lambda candidate: (
            int(candidate["min_rewrite_depth"]),
            str(candidate["candidate_id"]),
        )
    )
    budget_truncated = len(candidates) > max_candidates
    retained = candidates[:max_candidates]
    retained_ids = {candidate["candidate_id"] for candidate in retained}
    depth_growth = {
        str(depth): sum(
            int(candidate["min_rewrite_depth"]) <= depth
            for candidate in candidates
        )
        for depth in range(max_depth + 1)
    }
    rule_marginal = {
        rule.rule_id: sum(
            any(
                rule.rule_id in trace
                for trace in candidate["provenance_traces"]
            )
            for candidate in retained
        )
        for rule in mlp_rule_registry()
    }

    return {
        "schema_version": "bounded-rewrite-enumeration-v1",
        "family_id": "mlp_gate_up_activation_control",
        "max_rewrite_depth": max_depth,
        "max_fx_unique_candidates": max_candidates,
        "budget_truncated": budget_truncated,
        "num_enumerated_states": len(records),
        "num_valid_states": sum(len(group) for group in fingerprint_groups.values()),
        "num_invalid_states": len(invalid_states),
        "num_fx_unique_before_budget": len(candidates),
        "num_fx_unique_retained": len(retained),
        "valid_ratio": (
            sum(len(group) for group in fingerprint_groups.values())
            / len(records)
        ),
        "fx_retention": len(candidates) / len(records),
        "candidate_growth_by_depth": depth_growth,
        "rule_registry": [
            {
                "rule_id": rule.rule_id,
                "hypothesis": rule.hypothesis,
            }
            for rule in mlp_rule_registry()
        ],
        "rule_candidate_coverage": rule_marginal,
        "invalid_applications": dict(sorted(invalid.items())),
        "invalid_states": invalid_states,
        "enumeration_tree": [
            edge
            for edge in edges
            if any(
                edge["target_state_hash"]
                in candidate["provenance_state_hashes"]
                for candidate in retained
            )
        ],
        "fx_collapse_groups": fx_collapses,
        "candidates": retained,
        "retained_candidate_ids": sorted(retained_ids),
    }
