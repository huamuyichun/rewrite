from __future__ import annotations

import itertools
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from rewrite_selector.evaluation.execution_classes import (
    build_execution_classes,
    class_by_candidate,
)
from rewrite_selector.evaluation.statistics import percentile


def classify_pair(
    relative_deltas: list[float],
    noise_floor_relative: float,
) -> dict[str, Any]:
    per_session = [
        (
            "tie"
            if abs(delta) <= noise_floor_relative
            else ("first_faster" if delta > 0 else "second_faster")
        )
        for delta in relative_deltas
    ]
    non_ties = [label for label in per_session if label != "tie"]
    if not non_ties:
        label = "tie"
    elif len(set(non_ties)) == 1 and len(non_ties) >= 2:
        label = "strict"
    else:
        label = "ambiguous"
    reproducibility = (
        Counter(non_ties).most_common(1)[0][1] / len(non_ties)
        if non_ties
        else 1.0
    )
    return {
        "label": label,
        "per_session_labels": per_session,
        "relative_deltas": relative_deltas,
        "median_relative_delta": statistics.median(relative_deltas),
        "order_reproducibility": reproducibility,
    }


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_normalized_session(
    session_dir: Path,
    group_id: str,
) -> dict[str, Any]:
    result = _read(session_dir / "groups" / group_id / "result.json")
    resolved = _read(session_dir / "resolved_config.json")
    plans = resolved["rewrite_config"]["plans"]
    execution_classes = result.get("execution_classes")
    if execution_classes is None:
        execution_classes = build_execution_classes(
            plans,
            result["candidate_audits"],
        )
    mapping = class_by_candidate(execution_classes)
    profile = result["profile"]
    formal = result.get("formal_selection_unit", "candidate")
    if formal == "execution_class":
        class_summary = profile["candidate_summary"]
    else:
        candidate_summary = profile["candidate_summary"]
        class_summary = {
            execution_class["execution_class_id"]: candidate_summary[
                execution_class["canonical_candidate_id"]
            ]
            for execution_class in execution_classes
        }

    diagnostic = result.get("candidate_diagnostic_profile")
    candidate_summary = (
        diagnostic["candidate_summary"]
        if diagnostic is not None
        else profile["candidate_summary"]
    )
    candidate_summary = {
        candidate_id: summary
        for candidate_id, summary in candidate_summary.items()
        if candidate_id in mapping
    }
    fingerprints = {
        candidate_id: {
            "high_level": audit.get("high_level", {}).get("sha256"),
            "lowered": audit.get("lowered", {}).get("lowered_sha256"),
            "execution": audit.get("lowered", {}).get("execution_sha256"),
        }
        for candidate_id, audit in result["candidate_audits"].items()
        if audit.get("status") == "ok"
    }
    baseline_candidate = result["baseline_candidate_id"]
    baseline_class = mapping[baseline_candidate]
    candidate_winner = (
        min(
            candidate_summary,
            key=lambda key: float(candidate_summary[key]["p50_ms"]),
        )
        if candidate_summary
        else None
    )
    class_winner = min(
        class_summary,
        key=lambda key: float(class_summary[key]["p50_ms"]),
    )
    return {
        "session_id": session_dir.name,
        "session_dir": str(session_dir),
        "result": result,
        "execution_classes": execution_classes,
        "candidate_to_class": mapping,
        "class_summary": class_summary,
        "candidate_summary": candidate_summary,
        "fingerprints": fingerprints,
        "baseline_candidate_id": baseline_candidate,
        "baseline_class_id": baseline_class,
        "candidate_winner": candidate_winner,
        "class_winner": class_winner,
        "baseline_to_best_gain": (
            float(class_summary[baseline_class]["p50_ms"])
            / float(class_summary[class_winner]["p50_ms"])
            - 1.0
        ),
        "contaminated": bool(result.get("contaminated", False)),
        "contaminated_round_ratio": float(
            result.get("contaminated_round_ratio", 0.0)
        ),
        "monitor": profile.get("monitor", {}),
    }


def analyze_sessions(
    session_dirs: list[Path],
    group_id: str,
    noise_floor_relative: float = 0.02,
) -> dict[str, Any]:
    if len(session_dirs) < 2:
        raise ValueError("at least two sessions are required")
    sessions = [
        load_normalized_session(session_dir, group_id)
        for session_dir in session_dirs
    ]
    common_classes = sorted(
        set.intersection(
            *(set(session["class_summary"]) for session in sessions)
        )
    )
    common_candidates = sorted(
        set.intersection(
            *(set(session["fingerprints"]) for session in sessions)
        )
    )

    pairwise: list[dict[str, Any]] = []
    for first, second in itertools.combinations(common_classes, 2):
        deltas = [
            float(session["class_summary"][second]["p50_ms"])
            / float(session["class_summary"][first]["p50_ms"])
            - 1.0
            for session in sessions
        ]
        pairwise.append(
            {
                "first_execution_class_id": first,
                "second_execution_class_id": second,
                **classify_pair(deltas, noise_floor_relative),
            }
        )

    strict = [pair for pair in pairwise if pair["label"] == "strict"]
    fingerprint_stability = {
        candidate_id: {
            level: len(
                {
                    session["fingerprints"][candidate_id][level]
                    for session in sessions
                }
            )
            == 1
            for level in ("high_level", "lowered", "execution")
        }
        for candidate_id in common_candidates
    }
    mapping_stable = all(
        len(
            {
                session["candidate_to_class"].get(candidate_id)
                for session in sessions
            }
        )
        == 1
        for candidate_id in common_candidates
    )

    class_drift: dict[str, dict[str, float]] = {}
    for class_id in common_classes:
        values = [
            float(session["class_summary"][class_id]["p50_ms"])
            for session in sessions
        ]
        class_drift[class_id] = {
            "min_p50_ms": min(values),
            "max_p50_ms": max(values),
            "relative_range": max(values) / min(values) - 1.0,
            "median_p50_ms": statistics.median(values),
        }

    class_win_counts = Counter(
        session["class_winner"] for session in sessions
    )
    candidate_win_counts = Counter(
        session["candidate_winner"]
        for session in sessions
        if session["candidate_winner"] is not None
    )
    gains = [session["baseline_to_best_gain"] for session in sessions]

    class_table = []
    for session in sessions:
        for class_id in common_classes:
            class_table.append(
                {
                    "session_id": session["session_id"],
                    "execution_class_id": class_id,
                    **session["class_summary"][class_id],
                }
            )
    candidate_table = []
    for session in sessions:
        for candidate_id, summary in session["candidate_summary"].items():
            candidate_table.append(
                {
                    "session_id": session["session_id"],
                    "candidate_id": candidate_id,
                    "execution_class_id": session["candidate_to_class"][
                        candidate_id
                    ],
                    **summary,
                }
            )

    fixed_rows = []
    for class_id in common_classes:
        regrets = []
        for session in sessions:
            oracle = min(
                float(summary["p50_ms"])
                for summary in session["class_summary"].values()
            )
            selected = float(
                session["class_summary"][class_id]["p50_ms"]
            )
            regrets.append(selected / oracle - 1.0)
        fixed_rows.append(
            {
                "execution_class_id": class_id,
                "median_regret": statistics.median(regrets),
                "p90_regret": percentile(regrets, 0.9),
                "max_regret": max(regrets),
                "win_share": class_win_counts[class_id] / len(sessions),
                "regrets": regrets,
            }
        )
    best_fixed = min(
        fixed_rows,
        key=lambda row: (
            float(row["median_regret"]),
            float(row["max_regret"]),
            str(row["execution_class_id"]),
        ),
    )

    p3_p4_p5 = {}
    reference = sessions[0]["candidate_to_class"]
    if all(candidate in reference for candidate in (
        "p3_fused_chunk_silu",
        "p4_fused_split_silu_inplace",
        "p5_fused_chunk_manual_silu",
    )):
        p3_class = reference["p3_fused_chunk_silu"]
        p4_class = reference["p4_fused_split_silu_inplace"]
        p5_class = reference["p5_fused_chunk_manual_silu"]
        p3_p4_p5 = {
            "p3_execution_class_id": p3_class,
            "p4_execution_class_id": p4_class,
            "p5_execution_class_id": p5_class,
            "p3_p4_collapsed": p3_class == p4_class,
            "p3_p5_relative_deltas": [
                float(session["class_summary"][p5_class]["p50_ms"])
                / float(session["class_summary"][p3_class]["p50_ms"])
                - 1.0
                for session in sessions
            ],
        }
        p3_p4_p5["p3_p5_label"] = classify_pair(
            p3_p4_p5["p3_p5_relative_deltas"],
            noise_floor_relative,
        )["label"]

    return {
        "schema_version": "phase1-cross-session-analysis-v1",
        "group_id": group_id,
        "noise_floor_relative": noise_floor_relative,
        "session_ids": [session["session_id"] for session in sessions],
        "num_sessions": len(sessions),
        "all_sessions_clean": not any(
            session["contaminated"] for session in sessions
        ),
        "contaminated_round_ratio": statistics.mean(
            session["contaminated_round_ratio"] for session in sessions
        ),
        "class_table": class_table,
        "candidate_table": candidate_table,
        "execution_classes": sessions[0]["execution_classes"],
        "pairwise": pairwise,
        "strict_pair_order_reproducibility": (
            statistics.mean(
                float(pair["order_reproducibility"]) for pair in strict
            )
            if strict
            else 1.0
        ),
        "strict_pair_count": len(strict),
        "tie_pair_count": sum(
            pair["label"] == "tie" for pair in pairwise
        ),
        "ambiguous_pair_count": sum(
            pair["label"] == "ambiguous" for pair in pairwise
        ),
        "fingerprint_stability": fingerprint_stability,
        "all_fingerprints_stable": all(
            all(levels.values())
            for levels in fingerprint_stability.values()
        ),
        "execution_class_mapping_stable": mapping_stable,
        "candidate_winner_counts": dict(candidate_win_counts),
        "execution_class_winner_counts": dict(class_win_counts),
        "execution_class_winner_reproducibility": (
            class_win_counts.most_common(1)[0][1] / len(sessions)
        ),
        "baseline_to_best_gains": gains,
        "baseline_to_best_gain_median": statistics.median(gains),
        "baseline_to_best_gain_min": min(gains),
        "class_session_drift": class_drift,
        "fixed_class_rows": fixed_rows,
        "best_fixed_execution_class": best_fixed,
        "p3_p4_p5": p3_p4_p5,
    }
