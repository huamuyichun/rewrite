from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from rewrite_selector.evaluation.statistics import percentile, summarize_latency


NON_SEMANTIC_PLAN_FIELDS = {
    "candidate_id",
    "fx_sha256",
    "is_baseline",
    "min_rewrite_depth",
    "provenance_traces",
    "rewrite_family",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, value: Any, length: int = 16) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode()).hexdigest()
    return f"{prefix}_{digest[:length]}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_plan_record(plan: dict[str, Any], family_id: str) -> dict[str, Any]:
    parameters = {
        key: value
        for key, value in plan.items()
        if key not in NON_SEMANTIC_PLAN_FIELDS and key != "rewrite_trace"
    }
    trace = [str(rule_id) for rule_id in plan.get("rewrite_trace", [])]
    payload = {
        "family_id": family_id,
        "canonical_rewrite_trace": trace,
        "parameters": parameters,
    }
    label = "production_default" if plan.get("is_baseline") else "+".join(trace)
    return {
        "semantic_plan_id": stable_id("sem", payload),
        "label": label or "production_default",
        "family_id": family_id,
        "canonical_rewrite_trace": trace,
        "rule_ids": trace,
        "parameters": parameters,
        "is_production_default": bool(plan.get("is_baseline", False)),
        "semantic_payload": payload,
    }


def classify_relative_ci(
    ci_low: float,
    ci_high: float,
    noise_floor_relative: float,
) -> str:
    if ci_low > noise_floor_relative:
        return "first_faster"
    if ci_high < -noise_floor_relative:
        return "second_faster"
    if ci_low >= -noise_floor_relative and ci_high <= noise_floor_relative:
        return "tie"
    return "ambiguous"


def _round_values(profile: dict[str, Any], class_id: str) -> dict[int, list[float]]:
    values: dict[int, list[float]] = defaultdict(list)
    for row in profile.get("raw", []):
        if str(row["candidate_id"]) == class_id:
            values[int(row["round_index"])].append(float(row["latency_ms"]))
    return dict(values)


def _resampled_median(
    rounds: dict[int, list[float]],
    sampled_rounds: list[int],
) -> float:
    values = [
        latency
        for round_index in sampled_rounds
        for latency in rounds[round_index]
    ]
    return statistics.median(values)


def bootstrap_relative_difference_ci(
    session_pairs: list[tuple[dict[int, list[float]], dict[int, list[float]]]],
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if not session_pairs:
        raise ValueError("at least one paired session is required")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        selected_sessions = rng.choices(session_pairs, k=len(session_pairs))
        session_deltas: list[float] = []
        for first_rounds, second_rounds in selected_sessions:
            common_rounds = sorted(set(first_rounds) & set(second_rounds))
            if not common_rounds:
                raise ValueError("paired classes have no common blocked rounds")
            sampled_rounds = rng.choices(common_rounds, k=len(common_rounds))
            first = _resampled_median(first_rounds, sampled_rounds)
            second = _resampled_median(second_rounds, sampled_rounds)
            session_deltas.append(second / first - 1.0)
        estimates.append(statistics.median(session_deltas))
    alpha = (1.0 - confidence) / 2.0
    return percentile(estimates, alpha), percentile(estimates, 1.0 - alpha)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _normalize_driver(raw: str) -> str:
    versions = sorted({line.strip() for line in raw.splitlines() if line.strip()})
    return ",".join(versions)


def _resolve_record_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_registry(path: Path) -> list[dict[str, Any]]:
    entries = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        entry = json.loads(line)
        entry["registry_line_number"] = line_number
        entries.append(entry)
    return entries


def _environment_checks(
    environment: dict[str, Any],
    domain: dict[str, Any],
) -> dict[str, Any]:
    hardware = domain["hardware"]
    software = domain["software"]
    policy = domain["execution_policy"]
    gpu = environment.get("gpu", {})
    checks = {
        "python": str(environment.get("python", "")).startswith(software["python"]),
        "torch": environment.get("torch") == software["torch"],
        "cuda_runtime": environment.get("cuda_runtime") == software["cuda_runtime"],
        "driver": _normalize_driver(str(environment.get("driver", "")))
        == software["nvidia_driver"],
        "gpu_name": gpu.get("name") == hardware["gpu_name"],
        "gpu_uuid": gpu.get("uuid") == hardware["gpu_uuid"],
        "physical_index": gpu.get("physical_index") == hardware["physical_index"],
    }
    triton = environment.get("triton")
    cuda_device_order = environment.get("cuda_device_order")
    supplements = domain.get("historical_manifest_supplements", {})
    triton_source = "session_manifest"
    order_source = "session_manifest"
    if not triton:
        triton = supplements.get("triton", {}).get("value")
        triton_source = "frozen_domain_supplement"
    if not cuda_device_order:
        cuda_device_order = supplements.get("cuda_device_order", {}).get("value")
        order_source = "frozen_domain_supplement"
    checks["triton"] = triton == software["triton"]
    checks["cuda_device_order"] = cuda_device_order == policy["cuda_device_order"]
    return {
        "checks": checks,
        "all_match": all(checks.values()),
        "triton": {"value": triton, "binding_source": triton_source},
        "cuda_device_order": {
            "value": cuda_device_order,
            "binding_source": order_source,
        },
    }


def load_session(
    root: Path,
    entry: dict[str, Any],
    domain: dict[str, Any],
) -> dict[str, Any]:
    session_dir = _resolve_record_path(root, str(entry["artifact_path"]))
    required = {
        "environment": session_dir / "environment.json",
        "resolved_config": session_dir / "resolved_config.json",
        "session_summary": session_dir / "session_summary.json",
        "status": session_dir / "status.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing session artifacts: {missing}")
    environment = _read_json(required["environment"])
    resolved = _read_json(required["resolved_config"])
    session_summary = _read_json(required["session_summary"])
    status = _read_json(required["status"])
    result_paths = sorted((session_dir / "groups").glob("*/result.json"))
    if len(result_paths) != 1:
        raise ValueError(f"expected one group result in {session_dir}, got {len(result_paths)}")
    result_path = result_paths[0]
    result = _read_json(result_path)
    family_id = str(result["family_id"])
    plans = resolved["rewrite_config"]["plans"]
    semantic_plans = {
        str(plan["candidate_id"]): semantic_plan_record(plan, family_id)
        for plan in plans
    }
    candidate_to_semantic = {
        candidate_id: plan["semantic_plan_id"]
        for candidate_id, plan in semantic_plans.items()
    }
    semantic_plan_definitions = {
        plan["semantic_plan_id"]: plan for plan in semantic_plans.values()
    }
    class_by_id = {
        str(row["execution_class_id"]): row
        for row in result["execution_classes"]
    }
    class_signature_to_id: dict[tuple[str, ...], str] = {}
    class_records: dict[str, dict[str, Any]] = {}
    for class_id, row in class_by_id.items():
        semantic_ids = tuple(
            sorted(candidate_to_semantic[candidate_id] for candidate_id in row["candidate_ids"])
        )
        class_signature_to_id[semantic_ids] = class_id
        class_records[class_id] = {
            **row,
            "semantic_plan_ids": list(semantic_ids),
        }
    monitor = result["profile"].get("monitor", {})
    boundary = result["profile"].get("gpu_snapshots", [])
    environment_audit = _environment_checks(environment, domain)
    config_hash_matches = file_sha256(required["resolved_config"]) == entry["config_sha256"]
    source_matches = (
        entry.get("source_commit") == environment.get("source", {}).get("commit")
        and not bool(entry.get("source_dirty_at_run"))
        and not bool(environment.get("source", {}).get("dirty"))
    )
    cache_matches = (
        entry.get("cache_policy") == domain["execution_policy"]["cache_policy"]
        and not bool(entry.get("cache_preexisting"))
        and environment.get("cache", {}).get("policy")
        == domain["execution_policy"]["cache_policy"]
        and not bool(environment.get("cache", {}).get("preexisting"))
    )
    monitor_matches = (
        monitor.get("mode") == domain["execution_policy"]["timing_monitor_mode"]
        and monitor.get("backend")
        == domain["execution_policy"]["boundary_monitor_backend"]
        and int(monitor.get("sample_count", -1)) == 0
    )
    boundaries_match = bool(boundary) and all(
        snapshot.get("uuid") == domain["hardware"]["gpu_uuid"]
        and not snapshot.get("foreign_processes")
        for snapshot in boundary
    )
    status_ok = (
        entry.get("status") == "ok"
        and status.get("status") == "ok"
        and session_summary.get("status") == "ok"
        and not bool(result.get("contaminated"))
    )
    provenance_checks = {
        "status_ok": status_ok,
        "config_hash_matches": config_hash_matches,
        "source_matches": source_matches,
        "environment_domain_matches": environment_audit["all_match"],
        "cache_policy_matches": cache_matches,
        "monitor_policy_matches": monitor_matches,
        "boundary_snapshots_match": boundaries_match,
    }
    return {
        "run_id": str(entry["run_id"]),
        "session_id": str(entry["session_id"]),
        "session_dir": str(session_dir),
        "group_id": str(result["group_id"]),
        "result_path": str(result_path),
        "registry_entry": entry,
        "environment": environment,
        "environment_audit": environment_audit,
        "resolved_config": resolved,
        "result": result,
        "workload": result["workload"],
        "semantic_plan_definitions": semantic_plan_definitions,
        "candidate_to_semantic_plan": candidate_to_semantic,
        "class_records": class_records,
        "class_signature_to_id": class_signature_to_id,
        "provenance_checks": provenance_checks,
        "provenance_complete": all(provenance_checks.values()),
    }


def _summary_for_class(
    sessions: list[dict[str, Any]],
    signature: tuple[str, ...],
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    plan_definitions = sessions[0]["semantic_plan_definitions"]
    per_session = []
    schemas = set()
    execution_hashes = set()
    execution_ids = {}
    candidate_ids_by_session = {}
    total_samples = 0
    for session in sessions:
        class_id = session["class_signature_to_id"][signature]
        execution_ids[session["session_id"]] = class_id
        class_row = session["class_records"][class_id]
        candidate_ids_by_session[session["session_id"]] = list(class_row["candidate_ids"])
        execution_hashes.add(class_row.get("execution_sha256"))
        summaries = session["result"]["profile"]["candidate_summary"]
        summary = summaries[class_id]
        total_samples += int(summary["num_samples"])
        per_session.append({"session_id": session["session_id"], **summary})
        for candidate_id in class_row["candidate_ids"]:
            schemas.add(
                session["result"]["candidate_audits"][candidate_id]["lowered"].get(
                    "fingerprint_schema_version"
                )
            )
    p50_values = [float(row["p50_ms"]) for row in per_session]
    if len(per_session) == 1:
        aggregate = {
            key: per_session[0][key]
            for key in (
                "p50_ms",
                "mean_ms",
                "cv",
                "median_ci95_low_ms",
                "median_ci95_high_ms",
            )
        }
    else:
        aggregate = summarize_latency(p50_values, bootstrap_resamples, seed)
    return {
        "class_signature": stable_id("cls", list(signature)),
        "execution_class_id": next(iter(execution_ids.values())),
        "execution_class_ids_by_session": execution_ids,
        "candidate_ids": candidate_ids_by_session[per_session[0]["session_id"]],
        "candidate_ids_by_session": candidate_ids_by_session,
        "execution_sha256_values": sorted(value for value in execution_hashes if value),
        "fingerprint_schema_versions": sorted(value for value in schemas if value),
        "fingerprint_stable": len(execution_hashes) == 1 and len(schemas) == 1,
        "semantic_plan_ids": list(signature),
        "semantic_plans": [
            {
                "semantic_plan_id": plan_id,
                "label": plan_definitions[plan_id]["label"],
                "canonical_rewrite_trace": plan_definitions[plan_id][
                    "canonical_rewrite_trace"
                ],
                "parameters": plan_definitions[plan_id]["parameters"],
            }
            for plan_id in signature
        ],
        "raw_sample_count": total_samples,
        "num_sessions": len(per_session),
        "aggregation_unit": "raw_samples" if len(per_session) == 1 else "session_p50",
        **aggregate,
        "per_session": per_session,
    }


def _candidate_audit_summary(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_ids = sorted(sessions[0]["candidate_to_semantic_plan"])
    rows = []
    for candidate_id in candidate_ids:
        statuses = []
        for session in sessions:
            audit = session["result"]["candidate_audits"][candidate_id]
            statuses.append(
                {
                    "session_id": session["session_id"],
                    "audit": audit.get("status"),
                    "equivalence": audit.get("equivalence", {}).get("status"),
                    "eager": audit.get("equivalence", {}).get("eager", {}).get("status"),
                    "compiled": audit.get("equivalence", {})
                    .get("compiled", {})
                    .get("status"),
                    "alias": audit.get("equivalence", {}).get("alias", {}).get("status"),
                }
            )
        rows.append(
            {
                "candidate_id": candidate_id,
                "semantic_plan_id": sessions[0]["candidate_to_semantic_plan"][candidate_id],
                "all_ok": all(
                    all(value == "ok" for key, value in status.items() if key != "session_id")
                    for status in statuses
                ),
                "per_session": statuses,
            }
        )
    return rows


def _single_session_selection(
    session: dict[str, Any],
    signatures: list[tuple[str, ...]],
    noise_floor_relative: float,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    profile = session["result"]["profile"]
    class_rows = []
    for signature in signatures:
        class_id = session["class_signature_to_id"][signature]
        summary = profile["candidate_summary"][class_id]
        class_rows.append(
            {
                "signature": signature,
                "class_signature": stable_id("cls", list(signature)),
                "execution_class_id": class_id,
                "p50_ms": float(summary["p50_ms"]),
            }
        )
    by_signature = {row["signature"]: row for row in class_rows}
    pairwise = []
    for pair_index, (first_signature, second_signature) in enumerate(
        itertools.combinations(signatures, 2)
    ):
        first = by_signature[first_signature]
        second = by_signature[second_signature]
        first_rounds = _round_values(profile, first["execution_class_id"])
        second_rounds = _round_values(profile, second["execution_class_id"])
        ci_low, ci_high = bootstrap_relative_difference_ci(
            [(first_rounds, second_rounds)],
            bootstrap_resamples,
            seed + pair_index,
        )
        relative = second["p50_ms"] / first["p50_ms"] - 1.0
        pairwise.append(
            {
                "first_class_signature": first["class_signature"],
                "second_class_signature": second["class_signature"],
                "relative_difference": relative,
                "point_order": "first_faster" if relative > 0 else "second_faster",
                "relative_difference_ci95_low": ci_low,
                "relative_difference_ci95_high": ci_high,
                "preference": classify_relative_ci(
                    ci_low, ci_high, noise_floor_relative
                ),
            }
        )
    strictly_worse: set[str] = set()
    for pair in pairwise:
        if pair["preference"] == "first_faster":
            strictly_worse.add(pair["second_class_signature"])
        elif pair["preference"] == "second_faster":
            strictly_worse.add(pair["first_class_signature"])
    best_classes = sorted(
        row["class_signature"]
        for row in class_rows
        if row["class_signature"] not in strictly_worse
    )
    point_best = min(
        class_rows, key=lambda row: (row["p50_ms"], row["class_signature"])
    )
    production_plan = next(
        plan_id
        for plan_id, plan in session["semantic_plan_definitions"].items()
        if plan["is_production_default"]
    )
    production_signature = next(
        signature for signature in signatures if production_plan in signature
    )
    production = by_signature[production_signature]
    return {
        "run_id": session["run_id"],
        "session_id": session["session_id"],
        "point_best_class_signature": point_best["class_signature"],
        "point_best_execution_class_id": point_best["execution_class_id"],
        "noise_aware_best_class_signatures": best_classes,
        "baseline_to_point_oracle_gain": (
            production["p50_ms"] / point_best["p50_ms"] - 1.0
        ),
        "class_p50_ms": {
            row["class_signature"]: row["p50_ms"] for row in class_rows
        },
        "pairwise": pairwise,
        "contaminated": bool(session["result"].get("contaminated")),
    }


def summarize_session_reproducibility(
    session_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not session_rows:
        raise ValueError("at least one session row is required")
    num_sessions = len(session_rows)
    point_winners = Counter(
        str(row["point_best_class_signature"]) for row in session_rows
    )
    best_sets = Counter(
        tuple(row["noise_aware_best_class_signatures"]) for row in session_rows
    )
    class_signatures = sorted(session_rows[0]["class_p50_ms"])
    class_medians = {
        signature: statistics.median(
            float(row["class_p50_ms"][signature]) for row in session_rows
        )
        for signature in class_signatures
    }
    per_session_drift = []
    for row in session_rows:
        relative_shifts = [
            float(row["class_p50_ms"][signature]) / class_medians[signature] - 1.0
            for signature in class_signatures
        ]
        per_session_drift.append(
            {
                "session_id": row["session_id"],
                "median_relative_shift": statistics.median(relative_shifts),
                "max_absolute_class_shift": max(abs(value) for value in relative_shifts),
            }
        )
    class_ranges = [
        max(float(row["class_p50_ms"][signature]) for row in session_rows)
        / min(float(row["class_p50_ms"][signature]) for row in session_rows)
        - 1.0
        for signature in class_signatures
    ]
    pair_keys = [
        (
            pair["first_class_signature"],
            pair["second_class_signature"],
        )
        for pair in session_rows[0]["pairwise"]
    ]
    pair_rows = []
    for first_signature, second_signature in pair_keys:
        observations = []
        for session in session_rows:
            pair = next(
                item
                for item in session["pairwise"]
                if item["first_class_signature"] == first_signature
                and item["second_class_signature"] == second_signature
            )
            observations.append(
                {
                    "session_id": session["session_id"],
                    "point_order": pair["point_order"],
                    "preference": pair["preference"],
                    "relative_difference": pair["relative_difference"],
                }
            )
        point_orders = {row["point_order"] for row in observations}
        preferences = {row["preference"] for row in observations}
        pair_rows.append(
            {
                "first_class_signature": first_signature,
                "second_class_signature": second_signature,
                "point_order_reproducible": len(point_orders) == 1,
                "classification_reproducible": len(preferences) == 1,
                "strict_preference_reproduced": (
                    len(preferences) == 1
                    and next(iter(preferences))
                    in {"first_faster", "second_faster"}
                ),
                "per_session": observations,
            }
        )
    best_set_values = [set(row["noise_aware_best_class_signatures"]) for row in session_rows]
    gains = [float(row["baseline_to_point_oracle_gain"]) for row in session_rows]
    total_pairs = len(pair_rows)
    return {
        "num_sessions": num_sessions,
        "replication_target_met": num_sessions >= 3,
        "point_winner_reproducibility": max(point_winners.values()) / num_sessions,
        "point_winner_counts": dict(sorted(point_winners.items())),
        "best_set_exact_reproducibility": max(best_sets.values()) / num_sessions,
        "best_set_counts": [
            {"class_signatures": list(key), "count": value}
            for key, value in sorted(best_sets.items())
        ],
        "best_set_intersection": sorted(set.intersection(*best_set_values)),
        "best_set_union": sorted(set.union(*best_set_values)),
        "pairwise_point_order_reproducibility": (
            sum(row["point_order_reproducible"] for row in pair_rows) / total_pairs
            if total_pairs
            else 1.0
        ),
        "pairwise_classification_reproducibility": (
            sum(row["classification_reproducible"] for row in pair_rows) / total_pairs
            if total_pairs
            else 1.0
        ),
        "strict_preference_reproduced_pairs": sum(
            row["strict_preference_reproduced"] for row in pair_rows
        ),
        "total_pairwise_comparisons": total_pairs,
        "pairwise": pair_rows,
        "baseline_to_point_oracle_gain": {
            "min": min(gains),
            "median": statistics.median(gains),
            "max": max(gains),
            "values": gains,
        },
        "session_drift": {
            "median_class_p50_range": statistics.median(class_ranges),
            "max_class_p50_range": max(class_ranges),
            "per_session": per_session_drift,
        },
        "per_session": session_rows,
    }


def aggregate_group(
    sessions: list[dict[str, Any]],
    noise_floor_relative: float,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    if not sessions:
        raise ValueError("group must contain at least one session")
    sessions = sorted(sessions, key=lambda row: (row["run_id"], row["session_id"]))
    reference_signatures = set(sessions[0]["class_signature_to_id"])
    mapping_stable = all(
        set(session["class_signature_to_id"]) == reference_signatures
        for session in sessions
    )
    common_signatures = sorted(
        set.intersection(
            *(set(session["class_signature_to_id"]) for session in sessions)
        )
    )
    group_seed = int(hashlib.sha256(sessions[0]["group_id"].encode()).hexdigest()[:8], 16)
    class_rows = [
        _summary_for_class(
            sessions,
            signature,
            bootstrap_resamples,
            group_seed + index,
        )
        for index, signature in enumerate(common_signatures)
    ]
    class_by_signature = {
        tuple(row["semantic_plan_ids"]): row for row in class_rows
    }
    pairwise = []
    for pair_index, (first_signature, second_signature) in enumerate(
        itertools.combinations(common_signatures, 2)
    ):
        paired_rounds = []
        for session in sessions:
            first_id = session["class_signature_to_id"][first_signature]
            second_id = session["class_signature_to_id"][second_signature]
            profile = session["result"]["profile"]
            paired_rounds.append(
                (_round_values(profile, first_id), _round_values(profile, second_id))
            )
        first_row = class_by_signature[first_signature]
        second_row = class_by_signature[second_signature]
        observed = float(second_row["p50_ms"]) / float(first_row["p50_ms"]) - 1.0
        ci_low, ci_high = bootstrap_relative_difference_ci(
            paired_rounds,
            bootstrap_resamples,
            group_seed + 10_000 + pair_index,
        )
        preference = classify_relative_ci(ci_low, ci_high, noise_floor_relative)
        label = preference if preference in {"tie", "ambiguous"} else "strict"
        pairwise.append(
            {
                "first_class_signature": first_row["class_signature"],
                "first_execution_class_id": first_row["execution_class_id"],
                "second_class_signature": second_row["class_signature"],
                "second_execution_class_id": second_row["execution_class_id"],
                "relative_difference": observed,
                "relative_difference_ci95_low": ci_low,
                "relative_difference_ci95_high": ci_high,
                "noise_floor_relative": noise_floor_relative,
                "preference": preference,
                "label": label,
            }
        )
    strictly_worse: set[str] = set()
    for pair in pairwise:
        if pair["preference"] == "first_faster":
            strictly_worse.add(pair["second_class_signature"])
        elif pair["preference"] == "second_faster":
            strictly_worse.add(pair["first_class_signature"])
    best_classes = sorted(
        row["class_signature"]
        for row in class_rows
        if row["class_signature"] not in strictly_worse
    )
    best_class_set = set(best_classes)
    best_semantic_plans = sorted(
        semantic_plan_id
        for row in class_rows
        if row["class_signature"] in best_class_set
        for semantic_plan_id in row["semantic_plan_ids"]
    )
    point_best = min(class_rows, key=lambda row: (float(row["p50_ms"]), row["class_signature"]))
    point_worst = max(class_rows, key=lambda row: (float(row["p50_ms"]), row["class_signature"]))
    semantic_to_class = {
        semantic_plan_id: row["class_signature"]
        for row in class_rows
        for semantic_plan_id in row["semantic_plan_ids"]
    }
    plan_definitions = sessions[0]["semantic_plan_definitions"]
    production_plan = next(
        plan_id
        for plan_id, plan in plan_definitions.items()
        if plan["is_production_default"]
    )
    production_class = semantic_to_class[production_plan]
    production_row = next(
        row for row in class_rows if row["class_signature"] == production_class
    )
    oracle_latency = float(point_best["p50_ms"])
    production_gain = float(production_row["p50_ms"]) / oracle_latency - 1.0
    diagnostic_warnings = [
        {
            "session_id": session["session_id"],
            **warning,
        }
        for session in sessions
        for warning in session["result"].get("fingerprint_consistency", [])
        if warning.get("status") != "consistent"
    ]
    counts = {
        "enumerated": int(sessions[0]["result"]["num_requested_candidates"]),
        "valid": int(sessions[0]["result"]["num_valid_candidates"]),
        "fx_unique": int(sessions[0]["result"]["num_high_level_unique"]),
        "lowered_unique": int(sessions[0]["result"]["num_lowered_unique"]),
        "execution_unique": len(class_rows),
    }
    class_mapping_stable = mapping_stable and all(
        row["fingerprint_stable"] for row in class_rows
    )
    session_rows = [
        _single_session_selection(
            session,
            common_signatures,
            noise_floor_relative,
            bootstrap_resamples,
            group_seed + 20_000 + session_index * 1_000,
        )
        for session_index, session in enumerate(sessions)
    ]
    return {
        "group_id": sessions[0]["group_id"],
        "family_id": sessions[0]["result"]["family_id"],
        "workload": sessions[0]["workload"],
        "shape_bucket": (
            f'{sessions[0]["workload"]["phase"]}_bs'
            f'{sessions[0]["workload"]["batch_size"]}_s'
            f'{sessions[0]["workload"]["seq_len"]}'
        ),
        "session_ids": [session["session_id"] for session in sessions],
        "run_ids": sorted({session["run_id"] for session in sessions}),
        "num_sessions": len(sessions),
        "counts": counts,
        "execution_retention": counts["execution_unique"] / counts["fx_unique"],
        "candidate_audits": _candidate_audit_summary(sessions),
        "execution_classes": class_rows,
        "semantic_plan_to_execution_class": semantic_to_class,
        "pairwise": pairwise,
        "pair_counts": {
            label: sum(pair["label"] == label for pair in pairwise)
            for label in ("strict", "tie", "ambiguous")
        },
        "noise_aware_best_class_signatures": best_classes,
        "noise_aware_best_semantic_plan_ids": best_semantic_plans,
        "effective_competing_execution_classes": len(best_classes),
        "strict_semantic_winner": (
            best_semantic_plans[0] if len(best_semantic_plans) == 1 else None
        ),
        "point_best_class_signature": point_best["class_signature"],
        "point_best_execution_class_id": point_best["execution_class_id"],
        "point_best_p50_ms": point_best["p50_ms"],
        "point_worst_p50_ms": point_worst["p50_ms"],
        "best_worst_spread": float(point_worst["p50_ms"]) / oracle_latency - 1.0,
        "spread_exceeds_noise_floor": (
            float(point_worst["p50_ms"]) / oracle_latency - 1.0
            > noise_floor_relative
        ),
        "has_strict_preference": any(pair["label"] == "strict" for pair in pairwise),
        "production_semantic_plan_id": production_plan,
        "baseline_class_signature": production_class,
        "baseline_execution_class_id": production_row["execution_class_id"],
        "baseline_to_point_oracle_gain": production_gain,
        "production_to_noise_aware_oracle_gain": (
            0.0 if production_class in best_class_set else production_gain
        ),
        "point_oracle_p50_ms": oracle_latency,
        "raw_sample_count": sum(row["raw_sample_count"] for row in class_rows),
        "contaminated": any(bool(session["result"].get("contaminated")) for session in sessions),
        "contaminated_session_ratio": statistics.mean(
            bool(session["result"].get("contaminated")) for session in sessions
        ),
        "fingerprint_schema_versions": sorted(
            {
                schema
                for row in class_rows
                for schema in row["fingerprint_schema_versions"]
            }
        ),
        "fingerprint_stable": class_mapping_stable,
        "execution_class_mapping_stable": mapping_stable,
        "session_reproducibility": summarize_session_reproducibility(session_rows),
        "same_class_timing_diagnostic_warnings": diagnostic_warnings,
        "all_provenance_complete": all(session["provenance_complete"] for session in sessions),
    }


def _regret_summary(values: list[float]) -> dict[str, Any]:
    return {
        "p50": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "max": max(values),
        "values": values,
    }


def _plan_latency(group: dict[str, Any], semantic_plan_id: str) -> float:
    signature = group["semantic_plan_to_execution_class"][semantic_plan_id]
    row = next(
        row for row in group["execution_classes"] if row["class_signature"] == signature
    )
    return float(row["p50_ms"])


def _portfolio_metrics(
    groups: list[dict[str, Any]],
    semantic_plan_ids: tuple[str, ...],
) -> dict[str, Any]:
    raw_regrets = []
    noise_aware_regrets = []
    for group in groups:
        selected = min(_plan_latency(group, plan_id) for plan_id in semantic_plan_ids)
        oracle = float(group["point_oracle_p50_ms"])
        raw_regret = selected / oracle - 1.0
        selected_best = any(
            group["semantic_plan_to_execution_class"][plan_id]
            in group["noise_aware_best_class_signatures"]
            for plan_id in semantic_plan_ids
        )
        raw_regrets.append(raw_regret)
        noise_aware_regrets.append(0.0 if selected_best else raw_regret)
    return {
        "semantic_plan_ids": list(semantic_plan_ids),
        "raw_regret": _regret_summary(raw_regrets),
        "noise_aware_regret": _regret_summary(noise_aware_regrets),
    }


def _portfolio_key(row: dict[str, Any]) -> tuple[Any, ...]:
    regret = row["raw_regret"]
    return (
        float(regret["p50"]),
        float(regret["p90"]),
        float(regret["max"]),
        row["semantic_plan_ids"],
    )


def _rule_key(row: dict[str, Any]) -> tuple[Any, ...]:
    regret = row["raw_regret"]
    return (
        float(regret["p50"]),
        float(regret["p90"]),
        float(regret["max"]),
        str(row["rule"]),
    )


def aggregate_scope(
    name: str,
    groups: list[dict[str, Any]],
    semantic_plan_ids: list[str],
) -> dict[str, Any]:
    plan_rows = []
    for plan_id in semantic_plan_ids:
        portfolio = _portfolio_metrics(groups, (plan_id,))
        strict_wins = sum(group["strict_semantic_winner"] == plan_id for group in groups)
        possible_wins = sum(
            plan_id in group["noise_aware_best_semantic_plan_ids"] for group in groups
        )
        fractional_wins = sum(
            (
                1.0 / len(group["noise_aware_best_semantic_plan_ids"])
                if plan_id in group["noise_aware_best_semantic_plan_ids"]
                else 0.0
            )
            for group in groups
        )
        plan_rows.append(
            {
                "semantic_plan_id": plan_id,
                "strict_win_share": strict_wins / len(groups),
                "possible_win_share": possible_wins / len(groups),
                "fractional_tie_aware_win_share": fractional_wins / len(groups),
                **{key: value for key, value in portfolio.items() if key != "semantic_plan_ids"},
            }
        )
    plan_rows.sort(
        key=lambda row: (
            float(row["raw_regret"]["p50"]),
            float(row["raw_regret"]["p90"]),
            float(row["raw_regret"]["max"]),
            row["semantic_plan_id"],
        )
    )
    top_k = []
    for k in range(1, min(3, len(semantic_plan_ids)) + 1):
        portfolios = [
            _portfolio_metrics(groups, tuple(combo))
            for combo in itertools.combinations(semantic_plan_ids, k)
        ]
        top_k.append({"k": k, **min(portfolios, key=_portfolio_key)})
    fractional = [
        float(row["fractional_tie_aware_win_share"])
        for row in plan_rows
        if float(row["fractional_tie_aware_win_share"]) > 0
    ]
    entropy = -sum(value * math.log(value) for value in fractional)
    production_plan = groups[0]["production_semantic_plan_id"]
    production_row = next(row for row in plan_rows if row["semantic_plan_id"] == production_plan)
    raw_gains = [float(group["baseline_to_point_oracle_gain"]) for group in groups]
    noise_aware_gains = [
        float(group["production_to_noise_aware_oracle_gain"]) for group in groups
    ]
    pair_counts = {
        label: sum(group["pair_counts"][label] for group in groups)
        for label in ("strict", "tie", "ambiguous")
    }
    total_pairs = sum(pair_counts.values())
    return {
        "scope": name,
        "group_ids": [group["group_id"] for group in groups],
        "num_groups": len(groups),
        "execution_retention": {
            "values": [group["execution_retention"] for group in groups],
            "min": min(group["execution_retention"] for group in groups),
            "median": statistics.median(group["execution_retention"] for group in groups),
            "max": max(group["execution_retention"] for group in groups),
        },
        "pair_counts": pair_counts,
        "pair_ratios": {
            label: pair_counts[label] / total_pairs for label in pair_counts
        },
        "semantic_plan_rows": plan_rows,
        "best_fixed_semantic_plan": plan_rows[0],
        "production_default_semantic_plan": production_row,
        "production_to_point_oracle_gain": {
            "median": percentile(raw_gains, 0.5),
            "p90": percentile(raw_gains, 0.9),
            "max": max(raw_gains),
            "values": raw_gains,
        },
        "production_to_noise_aware_oracle_gain": {
            "median": percentile(noise_aware_gains, 0.5),
            "p90": percentile(noise_aware_gains, 0.9),
            "max": max(noise_aware_gains),
            "values": noise_aware_gains,
        },
        "top_k_oracle_curves": top_k,
        "winner_entropy": entropy,
        "winner_entropy_normalized": (
            entropy / math.log(len(fractional)) if len(fractional) > 1 else 0.0
        ),
        "average_noise_aware_best_class_count": statistics.fmean(
            group["effective_competing_execution_classes"] for group in groups
        ),
        "average_noise_aware_best_semantic_plan_count": statistics.fmean(
            len(group["noise_aware_best_semantic_plan_ids"]) for group in groups
        ),
        "groups_with_strict_preference": sum(
            group["has_strict_preference"] for group in groups
        ),
        "groups_with_spread_over_noise_floor": sum(
            group["spread_exceeds_noise_floor"] for group in groups
        ),
    }


def _partition_policy(
    name: str,
    groups: list[dict[str, Any]],
    semantic_plan_ids: list[str],
    key_function: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        buckets[key_function(group)].append(group)
    selections = {}
    for bucket, members in sorted(buckets.items()):
        rows = [_portfolio_metrics(members, (plan_id,)) for plan_id in semantic_plan_ids]
        selections[bucket] = min(rows, key=_portfolio_key)["semantic_plan_ids"][0]
    raw_regrets = []
    noise_aware_regrets = []
    for group in groups:
        plan_id = selections[key_function(group)]
        selected = _plan_latency(group, plan_id)
        oracle = float(group["point_oracle_p50_ms"])
        raw = selected / oracle - 1.0
        signature = group["semantic_plan_to_execution_class"][plan_id]
        raw_regrets.append(raw)
        noise_aware_regrets.append(
            0.0 if signature in group["noise_aware_best_class_signatures"] else raw
        )
    return {
        "rule": name,
        "num_buckets": len(buckets),
        "bucket_selections": selections,
        "raw_regret": _regret_summary(raw_regrets),
        "noise_aware_regret": _regret_summary(noise_aware_regrets),
    }


def simple_rule_diagnostics(
    groups: list[dict[str, Any]],
    semantic_plan_ids: list[str],
) -> list[dict[str, Any]]:
    diagnostics = [
        _partition_policy("global_fixed", groups, semantic_plan_ids, lambda group: "all"),
        _partition_policy(
            "decode_vs_prefill",
            groups,
            semantic_plan_ids,
            lambda group: str(group["workload"]["phase"]),
        ),
        _partition_policy(
            "context_type",
            groups,
            semantic_plan_ids,
            lambda group: str(group["workload"].get("context", "none")),
        ),
        _partition_policy(
            "exact_shape_bucket",
            groups,
            semantic_plan_ids,
            lambda group: str(group["shape_bucket"]),
        ),
    ]
    for field, rule_name in (
        ("batch_size", "batch_threshold"),
        ("seq_len", "sequence_length_threshold"),
    ):
        values = sorted({int(group["workload"][field]) for group in groups})
        threshold_rows = []
        for threshold in values[:-1]:
            row = _partition_policy(
                f"{rule_name}<={threshold}",
                groups,
                semantic_plan_ids,
                lambda group, field=field, threshold=threshold: (
                    "low" if int(group["workload"][field]) <= threshold else "high"
                ),
            )
            threshold_rows.append(row)
        if threshold_rows:
            diagnostics.append(min(threshold_rows, key=_rule_key))
    token_values = sorted(
        {
            int(group["workload"]["batch_size"]) * int(group["workload"]["seq_len"])
            for group in groups
        }
    )
    token_rows = []
    for threshold in token_values[:-1]:
        token_rows.append(
            _partition_policy(
                f"token_threshold<={threshold}",
                groups,
                semantic_plan_ids,
                lambda group, threshold=threshold: (
                    "low"
                    if int(group["workload"]["batch_size"])
                    * int(group["workload"]["seq_len"])
                    <= threshold
                    else "high"
                ),
            )
        )
    if token_rows:
        diagnostics.append(min(token_rows, key=_rule_key))
    return diagnostics


def analyze_registry_sessions(
    root: Path,
    registry_path: Path,
    domain_path: Path,
    run_ids: set[str],
    noise_floor_relative: float = 0.02,
    bootstrap_resamples: int = 2000,
) -> dict[str, Any]:
    domain = _read_json(domain_path)
    entries = [
        entry
        for entry in load_registry(registry_path)
        if str(entry.get("run_id")) in run_ids
    ]
    if not entries:
        raise ValueError(f"no registry entries found for run ids: {sorted(run_ids)}")
    sessions = [load_session(root, entry, domain) for entry in entries]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        grouped[session["group_id"]].append(session)
    groups = [
        aggregate_group(members, noise_floor_relative, bootstrap_resamples)
        for _, members in sorted(grouped.items())
    ]
    semantic_plan_definitions = sessions[0]["semantic_plan_definitions"]
    semantic_plan_ids = sorted(semantic_plan_definitions)
    scopes = {
        "all": aggregate_scope("all", groups, semantic_plan_ids),
        "decode": aggregate_scope(
            "decode",
            [group for group in groups if group["workload"]["phase"] == "decode"],
            semantic_plan_ids,
        ),
        "prefill": aggregate_scope(
            "prefill",
            [group for group in groups if group["workload"]["phase"] == "prefill"],
            semantic_plan_ids,
        ),
    }
    by_context = {
        context: aggregate_scope(
            f"context:{context}",
            [
                group
                for group in groups
                if str(group["workload"].get("context", "none")) == context
            ],
            semantic_plan_ids,
        )
        for context in sorted(
            {str(group["workload"].get("context", "none")) for group in groups}
        )
    }
    by_shape_bucket = {
        bucket: aggregate_scope(
            f"shape:{bucket}",
            [group for group in groups if group["shape_bucket"] == bucket],
            semantic_plan_ids,
        )
        for bucket in sorted({str(group["shape_bucket"]) for group in groups})
    }
    global_top_k = scopes["all"]["top_k_oracle_curves"]
    for group in groups:
        group["top_k_oracle"] = []
        for row in global_top_k:
            portfolio = _portfolio_metrics([group], tuple(row["semantic_plan_ids"]))
            group["top_k_oracle"].append({"k": row["k"], **portfolio})
    diagnostics = simple_rule_diagnostics(groups, semantic_plan_ids)
    all_scope = scopes["all"]
    best_fixed_id = all_scope["best_fixed_semantic_plan"]["semantic_plan_id"]
    strict_winner_plans = sorted(
        {
            group["strict_semantic_winner"]
            for group in groups
            if group["strict_semantic_winner"] is not None
        }
    )
    always_possible_plans = sorted(
        set.intersection(
            *(set(group["noise_aware_best_semantic_plan_ids"]) for group in groups)
        )
    )
    questions = {
        "retention_stable_8_to_6": all(
            group["counts"]["fx_unique"] == 8
            and group["counts"]["execution_unique"] == 6
            for group in groups
        ),
        "strictly_distinguishable_class_counts": [
            len(
                {
                    class_signature
                    for pair in group["pairwise"]
                    if pair["label"] == "strict"
                    for class_signature in (
                        pair["first_class_signature"],
                        pair["second_class_signature"],
                    )
                }
            )
            for group in groups
        ],
        "groups_with_spread_over_noise_floor": all_scope[
            "groups_with_spread_over_noise_floor"
        ],
        "groups_with_strict_preference": all_scope["groups_with_strict_preference"],
        "average_noise_aware_best_set_size": all_scope[
            "average_noise_aware_best_class_count"
        ],
        "winner_semantic_plan_sets_vary": len(
            {tuple(group["noise_aware_best_semantic_plan_ids"]) for group in groups}
        )
        > 1,
        "always_possible_semantic_plan_ids": always_possible_plans,
        "strict_winner_semantic_plan_ids": strict_winner_plans,
        "global_best_fixed_semantic_plan_id": best_fixed_id,
        "global_best_fixed_possible_win_share": all_scope[
            "best_fixed_semantic_plan"
        ]["possible_win_share"],
        "global_best_fixed_regret": {
            "raw": all_scope["best_fixed_semantic_plan"]["raw_regret"],
            "noise_aware": all_scope["best_fixed_semantic_plan"][
                "noise_aware_regret"
            ],
        },
        "production_to_point_oracle_gain": all_scope[
            "production_to_point_oracle_gain"
        ],
        "production_to_noise_aware_oracle_gain": all_scope[
            "production_to_noise_aware_oracle_gain"
        ],
        "preliminary_context_sensitive_value": (
            len(strict_winner_plans) >= 2
            and float(
                all_scope["best_fixed_semantic_plan"]["raw_regret"]["p90"]
            )
            > noise_floor_relative
        ),
    }
    session_audits = [
        {
            "run_id": session["run_id"],
            "session_id": session["session_id"],
            "group_id": session["group_id"],
            "source_commit": session["registry_entry"]["source_commit"],
            "config_sha256": session["registry_entry"]["config_sha256"],
            "hardware_environment_domain_id": domain["domain_id"],
            "gpu_uuid": session["environment"].get("gpu", {}).get("uuid"),
            "driver": _normalize_driver(str(session["environment"].get("driver", ""))),
            "cuda_runtime": session["environment"].get("cuda_runtime"),
            "torch": session["environment"].get("torch"),
            "triton": session["environment_audit"]["triton"],
            "cache_policy": session["registry_entry"].get("cache_policy"),
            "monitor_policy": session["result"]["profile"].get("monitor"),
            "provenance_checks": session["provenance_checks"],
            "provenance_complete": session["provenance_complete"],
        }
        for session in sessions
    ]
    return {
        "schema_version": "phase2-family-discovery-analysis-v1",
        "family_id": groups[0]["family_id"],
        "hardware_environment_domain": domain,
        "absolute_latency_scope": domain["latency_policy"][
            "absolute_latency_merge_scope"
        ],
        "noise_floor_relative": noise_floor_relative,
        "bootstrap_resamples": bootstrap_resamples,
        "run_ids": sorted(run_ids),
        "session_audits": session_audits,
        "all_session_provenance_complete": all(
            row["provenance_complete"] for row in session_audits
        ),
        "semantic_plan_definitions": semantic_plan_definitions,
        "groups": groups,
        "scopes": scopes,
        "by_context": by_context,
        "by_shape_bucket": by_shape_bucket,
        "simple_rule_diagnostics": diagnostics,
        "questions": questions,
    }
