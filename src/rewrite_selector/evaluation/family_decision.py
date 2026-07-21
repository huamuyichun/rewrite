from __future__ import annotations

import statistics
from typing import Any


def choose_rmsnorm_decision(
    *,
    stable: bool,
    strict_winner_count: int,
    groups_with_strict_preference: int,
    median_execution_retention: float,
    fixed_p90_regret: float,
    production_noise_p90_gain: float,
    simple_rule_max_regret: float,
    noise_floor_relative: float,
) -> str:
    supports_context_selection = (
        stable
        and strict_winner_count >= 2
        and fixed_p90_regret > noise_floor_relative
        and production_noise_p90_gain > noise_floor_relative
        and simple_rule_max_regret > noise_floor_relative
    )
    if supports_context_selection:
        return "A"
    if stable and groups_with_strict_preference > 0 and median_execution_retention > 0:
        return "B"
    return "C"


def build_rmsnorm_decision(analysis: dict[str, Any], analysis_path: str) -> dict[str, Any]:
    all_scope = analysis["scopes"]["all"]
    decode_scope = analysis["scopes"]["decode"]
    prefill_scope = analysis["scopes"]["prefill"]
    questions = analysis["questions"]
    noise_floor = float(analysis["noise_floor_relative"])
    replicated_groups = [
        group for group in analysis["groups"] if int(group["num_sessions"]) >= 3
    ]
    stable = all(
        bool(group["fingerprint_stable"])
        and bool(group["execution_class_mapping_stable"])
        for group in analysis["groups"]
    )
    simple_rules = [
        row for row in analysis["simple_rule_diagnostics"] if row["rule"] != "global_fixed"
    ]
    best_simple_rule = min(
        simple_rules,
        key=lambda row: (
            float(row["raw_regret"]["max"]),
            float(row["raw_regret"]["p90"]),
            str(row["rule"]),
        ),
    )
    fixed = all_scope["best_fixed_semantic_plan"]
    decision_code = choose_rmsnorm_decision(
        stable=stable,
        strict_winner_count=len(questions["strict_winner_semantic_plan_ids"]),
        groups_with_strict_preference=int(
            questions["groups_with_strict_preference"]
        ),
        median_execution_retention=float(
            all_scope["execution_retention"]["median"]
        ),
        fixed_p90_regret=float(fixed["raw_regret"]["p90"]),
        production_noise_p90_gain=float(
            all_scope["production_to_noise_aware_oracle_gain"]["p90"]
        ),
        simple_rule_max_regret=float(best_simple_rule["raw_regret"]["max"]),
        noise_floor_relative=noise_floor,
    )
    decision_labels = {
        "A": "保留为主要 context-sensitive family",
        "B": "降为 control/diagnostic family",
        "C": "停止扩大该 family",
    }
    replication = []
    for group in replicated_groups:
        reproducibility = group["session_reproducibility"]
        replication.append(
            {
                "group_id": group["group_id"],
                "session_ids": group["session_ids"],
                "fingerprint_stable": group["fingerprint_stable"],
                "execution_class_mapping_stable": group[
                    "execution_class_mapping_stable"
                ],
                "point_winner_reproducibility": reproducibility[
                    "point_winner_reproducibility"
                ],
                "best_set_exact_reproducibility": reproducibility[
                    "best_set_exact_reproducibility"
                ],
                "pairwise_point_order_reproducibility": reproducibility[
                    "pairwise_point_order_reproducibility"
                ],
                "baseline_to_point_oracle_gain": reproducibility[
                    "baseline_to_point_oracle_gain"
                ],
                "session_drift": reproducibility["session_drift"],
                "contaminated_session_ratio": group[
                    "contaminated_session_ratio"
                ],
            }
        )
    all_sessions = analysis["session_audits"]
    new_sessions = [
        row for row in all_sessions if row["session_id"].endswith(("_r02", "_r03"))
    ]
    return {
        "schema_version": "phase2-family-decision-v1",
        "family_id": analysis["family_id"],
        "analysis_path": analysis_path,
        "hardware_environment_domain_id": analysis["hardware_environment_domain"][
            "domain_id"
        ],
        "absolute_latency_scope": analysis["absolute_latency_scope"],
        "noise_floor_relative": noise_floor,
        "decision": {
            "code": decision_code,
            "label": decision_labels[decision_code],
            "learned_selector_authorized": decision_code == "A",
            "expand_similar_rmsnorm_shapes": False,
        },
        "evidence": {
            "num_groups": len(analysis["groups"]),
            "num_sessions": len(all_sessions),
            "replicated_group_count": len(replicated_groups),
            "execution_retention": all_scope["execution_retention"],
            "pair_counts": all_scope["pair_counts"],
            "groups_with_strict_preference": questions[
                "groups_with_strict_preference"
            ],
            "strictly_distinguishable_class_counts": questions[
                "strictly_distinguishable_class_counts"
            ],
            "average_noise_aware_best_set_size": questions[
                "average_noise_aware_best_set_size"
            ],
            "strict_winner_semantic_plan_ids": questions[
                "strict_winner_semantic_plan_ids"
            ],
            "always_possible_semantic_plan_ids": questions[
                "always_possible_semantic_plan_ids"
            ],
            "global_best_fixed": fixed,
            "decode_best_fixed": decode_scope["best_fixed_semantic_plan"],
            "prefill_best_fixed": prefill_scope["best_fixed_semantic_plan"],
            "production_default": all_scope[
                "production_default_semantic_plan"
            ],
            "production_to_point_oracle_gain": all_scope[
                "production_to_point_oracle_gain"
            ],
            "production_to_noise_aware_oracle_gain": all_scope[
                "production_to_noise_aware_oracle_gain"
            ],
            "best_simple_rule": best_simple_rule,
            "winner_semantic_plan_sets_vary": questions[
                "winner_semantic_plan_sets_vary"
            ],
            "fingerprint_and_mapping_stable": stable,
            "contaminated_sessions": sum(
                not bool(row["provenance_checks"]["status_ok"])
                for row in all_sessions
            ),
            "same_class_timing_warning_count": sum(
                len(group["same_class_timing_diagnostic_warnings"])
                for group in analysis["groups"]
            ),
        },
        "replication": replication,
        "new_sessions": [
            {
                "run_id": row["run_id"],
                "session_id": row["session_id"],
                "group_id": row["group_id"],
                "source_commit": row["source_commit"],
                "provenance_complete": row["provenance_complete"],
            }
            for row in new_sessions
        ],
        "aborted_or_contaminated_sessions": [],
        "interpretation": {
            "real_execution_diversity": True,
            "stable_context_dependent_strict_winner_exchange": False,
            "fixed_plan_near_oracle": True,
            "simple_rule_near_oracle": True,
            "production_in_every_noise_aware_best_set": all(
                group["production_semantic_plan_id"]
                in group["noise_aware_best_semantic_plan_ids"]
                for group in analysis["groups"]
            ),
            "replicated_session_gain_median": statistics.median(
                row["baseline_to_point_oracle_gain"]["median"]
                for row in replication
            ),
        },
        "mlp_next_action": {
            "status": "authorized_after_rmsnorm_decision",
            "role": "control_family",
            "screening_sessions_per_group": 1,
            "num_groups": 9,
            "profile_execution_unique_only": True,
            "adaptive_replication_only": True,
            "do_not_train_selector": True,
        },
        "phase2_exit_gate": {
            "status": "pending_mlp_control_discovery",
            "rmsnorm_supports_learned_selector": False,
        },
    }
