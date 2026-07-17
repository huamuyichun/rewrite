from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def bootstrap_median_ci(
    values: Sequence[float],
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates = [
        statistics.median(rng.choices(list(values), k=len(values)))
        for _ in range(resamples)
    ]
    alpha = (1 - confidence) / 2
    return percentile(estimates, alpha), percentile(estimates, 1 - alpha)


def summarize_latency(values: Sequence[float], resamples: int, seed: int) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    ci_low, ci_high = bootstrap_median_ci(values, resamples, seed)
    return {
        "num_samples": len(values),
        "p50_ms": percentile(values, 0.5),
        "p10_ms": percentile(values, 0.1),
        "p90_ms": percentile(values, 0.9),
        "mean_ms": mean,
        "std_ms": std,
        "cv": std / mean if mean > 0 else float("nan"),
        "median_ci95_low_ms": ci_low,
        "median_ci95_high_ms": ci_high,
    }


def relative_regret(selected_latency: float, oracle_latency: float) -> float:
    if oracle_latency <= 0:
        raise ValueError("oracle latency must be positive")
    return selected_latency / oracle_latency - 1.0

