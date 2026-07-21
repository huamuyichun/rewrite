import pytest

from rewrite_selector.evaluation.phase2_analysis import (
    aggregate_scope,
    bootstrap_relative_difference_ci,
    classify_relative_ci,
    semantic_plan_record,
    summarize_session_reproducibility,
)


def test_semantic_plan_id_ignores_candidate_identity() -> None:
    first = {
        "candidate_id": "local-a",
        "fx_sha256": "aaa",
        "is_baseline": False,
        "rewrite_family": "rmsnorm_residual_boundary",
        "rewrite_trace": ["rmsnorm.decompose_square_mul"],
        "implementation": "square_mul",
        "flatten": False,
        "scale_association": "left",
    }
    second = {**first, "candidate_id": "local-b", "fx_sha256": "bbb"}
    first_record = semantic_plan_record(first, "rmsnorm_residual_boundary")
    second_record = semantic_plan_record(second, "rmsnorm_residual_boundary")
    assert first_record["semantic_plan_id"] == second_record["semantic_plan_id"]

    changed = {**second, "scale_association": "right"}
    changed_record = semantic_plan_record(changed, "rmsnorm_residual_boundary")
    assert changed_record["semantic_plan_id"] != first_record["semantic_plan_id"]


def test_relative_ci_classifier_requires_ci_to_clear_noise_floor() -> None:
    assert classify_relative_ci(0.021, 0.05, 0.02) == "first_faster"
    assert classify_relative_ci(-0.05, -0.021, 0.02) == "second_faster"
    assert classify_relative_ci(-0.01, 0.015, 0.02) == "tie"
    assert classify_relative_ci(0.01, 0.03, 0.02) == "ambiguous"


def test_paired_round_bootstrap_detects_large_relative_difference() -> None:
    first = {index: [1.0, 1.0] for index in range(10)}
    second = {index: [1.05, 1.05] for index in range(10)}
    low, high = bootstrap_relative_difference_ci(
        [(first, second)],
        resamples=200,
        seed=7,
    )
    assert low == pytest.approx(0.05)
    assert high == pytest.approx(0.05)
    assert classify_relative_ci(low, high, 0.02) == "first_faster"


def _group(
    group_id: str,
    first_latency: float,
    second_latency: float,
    best_plans: list[str],
) -> dict:
    first_signature = f"{group_id}-first"
    second_signature = f"{group_id}-second"
    best_signatures = {
        first_signature if plan_id == "sem_a" else second_signature
        for plan_id in best_plans
    }
    return {
        "group_id": group_id,
        "execution_classes": [
            {"class_signature": first_signature, "p50_ms": first_latency},
            {"class_signature": second_signature, "p50_ms": second_latency},
        ],
        "semantic_plan_to_execution_class": {
            "sem_a": first_signature,
            "sem_b": second_signature,
        },
        "point_oracle_p50_ms": min(first_latency, second_latency),
        "noise_aware_best_class_signatures": sorted(best_signatures),
        "noise_aware_best_semantic_plan_ids": best_plans,
        "strict_semantic_winner": best_plans[0] if len(best_plans) == 1 else None,
        "production_semantic_plan_id": "sem_a",
        "production_to_noise_aware_oracle_gain": 0.0,
        "baseline_to_point_oracle_gain": first_latency / min(first_latency, second_latency)
        - 1.0,
        "pair_counts": {"strict": 1, "tie": 0, "ambiguous": 0},
        "execution_retention": 1.0,
        "effective_competing_execution_classes": len(best_signatures),
        "has_strict_preference": True,
        "spread_exceeds_noise_floor": True,
    }


def test_fixed_plan_uses_semantic_plan_across_group_local_classes() -> None:
    groups = [
        _group("g1", 1.0, 1.10, ["sem_a"]),
        _group("g2", 1.05, 1.0, ["sem_b"]),
    ]
    scope = aggregate_scope("all", groups, ["sem_a", "sem_b"])
    assert scope["best_fixed_semantic_plan"]["semantic_plan_id"] == "sem_a"
    assert scope["best_fixed_semantic_plan"]["raw_regret"]["max"] == pytest.approx(0.05)
    assert scope["semantic_plan_rows"][0]["possible_win_share"] == pytest.approx(0.5)


def test_session_reproducibility_reports_winner_order_gain_and_drift() -> None:
    def session(
        session_id: str,
        first: float,
        second: float,
        best: list[str],
        preference: str,
    ) -> dict:
        return {
            "session_id": session_id,
            "point_best_class_signature": "cls_a" if first < second else "cls_b",
            "noise_aware_best_class_signatures": best,
            "baseline_to_point_oracle_gain": first / min(first, second) - 1.0,
            "class_p50_ms": {"cls_a": first, "cls_b": second},
            "pairwise": [
                {
                    "first_class_signature": "cls_a",
                    "second_class_signature": "cls_b",
                    "point_order": "first_faster" if first < second else "second_faster",
                    "preference": preference,
                    "relative_difference": second / first - 1.0,
                }
            ],
        }

    summary = summarize_session_reproducibility(
        [
            session("s1", 1.0, 1.03, ["cls_a"], "first_faster"),
            session("s2", 1.01, 1.02, ["cls_a", "cls_b"], "tie"),
            session("s3", 1.02, 1.01, ["cls_a", "cls_b"], "ambiguous"),
        ]
    )
    assert summary["replication_target_met"] is True
    assert summary["point_winner_reproducibility"] == pytest.approx(2 / 3)
    assert summary["best_set_exact_reproducibility"] == pytest.approx(2 / 3)
    assert summary["best_set_intersection"] == ["cls_a"]
    assert summary["pairwise_point_order_reproducibility"] == 0.0
    assert summary["pairwise_classification_reproducibility"] == 0.0
    assert summary["baseline_to_point_oracle_gain"]["max"] == pytest.approx(
        1.02 / 1.01 - 1.0
    )
    assert summary["session_drift"]["max_class_p50_range"] == pytest.approx(0.02)
