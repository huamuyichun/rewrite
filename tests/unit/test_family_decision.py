from rewrite_selector.evaluation.family_decision import choose_rmsnorm_decision


def test_context_sensitive_decision_requires_all_value_gates() -> None:
    common = {
        "stable": True,
        "strict_winner_count": 2,
        "groups_with_strict_preference": 8,
        "median_execution_retention": 0.75,
        "fixed_p90_regret": 0.04,
        "production_noise_p90_gain": 0.03,
        "simple_rule_max_regret": 0.03,
        "noise_floor_relative": 0.02,
    }
    assert choose_rmsnorm_decision(**common) == "A"
    assert choose_rmsnorm_decision(**{**common, "fixed_p90_regret": 0.01}) == "B"
    assert choose_rmsnorm_decision(**{**common, "stable": False}) == "C"
