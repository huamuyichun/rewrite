from rewrite_selector.evaluation.phase1_analysis import classify_pair


def test_pair_classifier_handles_strict_tie_and_ambiguous() -> None:
    strict = classify_pair([0.04, 0.05, 0.03], 0.02)
    assert strict["label"] == "strict"
    assert strict["order_reproducibility"] == 1.0

    tie = classify_pair([0.001, -0.01, 0.015], 0.02)
    assert tie["label"] == "tie"

    ambiguous = classify_pair([0.04, -0.05, 0.03], 0.02)
    assert ambiguous["label"] == "ambiguous"
    assert ambiguous["order_reproducibility"] < 1.0
