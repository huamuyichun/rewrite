import pytest

from rewrite_selector.evaluation.statistics import (
    bootstrap_median_ci,
    percentile,
    relative_regret,
    summarize_latency,
)


def test_percentile_and_summary() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.5) == 2.5
    summary = summarize_latency(values, resamples=200, seed=3)
    assert summary["p50_ms"] == 2.5
    assert summary["num_samples"] == 4
    low, high = bootstrap_median_ci(values, resamples=200, seed=3)
    assert low <= 2.5 <= high


def test_relative_regret() -> None:
    assert relative_regret(1.05, 1.0) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        relative_regret(1.0, 0.0)

