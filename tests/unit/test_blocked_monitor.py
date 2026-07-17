import torch

from rewrite_selector.profiling.blocked import run_blocked_rounds


def test_monitor_off_records_rounds_without_polls(monkeypatch) -> None:
    monkeypatch.setattr(
        "rewrite_selector.profiling.blocked.gpu_snapshot",
        lambda backend: {
            "timestamp_ns": 0,
            "foreign_processes": [],
            "backend": backend,
        },
    )
    example = torch.ones(2)
    result = run_blocked_rounds(
        {"identity": lambda value: value + 1},
        example,
        rounds=2,
        warmup_per_round=0,
        samples_per_round=1,
        iterations_per_sample=1,
        randomization_seed=1,
        bootstrap_resamples=20,
        monitor_mode="off",
        monitor_backend="nvml",
        monitor_interval_seconds=1.0,
    )
    assert len(result["rounds"]) == 2
    assert len(result["raw"]) == 2
    assert result["monitor"]["mode"] == "off"
    assert result["monitor"]["sample_count"] == 0
    assert result["contaminated_round_ratio"] == 0.0
