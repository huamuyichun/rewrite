import json
from pathlib import Path

import pytest
import torch

from rewrite_selector.ir.families import get_family_adapter
from rewrite_selector.rewrites.registry import resolve_rewrite_config
from scripts.run_phase1_audit import profile_workload


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("config_name", "workload_value", "expected_count"),
    [
        (
            "mlp_bounded_v1.json",
            {
                "group_id": "mlp_runner_test",
                "model_id": "tiny",
                "phase": "decode",
                "batch_size": 1,
                "seq_len": 1,
                "hidden_dim": 8,
                "intermediate_dim": 16,
                "dtype": "fp32",
                "seed": 7,
            },
            19,
        ),
        (
            "rmsnorm_bounded_v1.json",
            {
                "group_id": "rmsnorm_runner_test",
                "model_id": "tiny",
                "phase": "decode",
                "batch_size": 1,
                "seq_len": 1,
                "hidden_dim": 16,
                "dtype": "fp32",
                "seed": 7,
                "context": "norm_only",
            },
            8,
        ),
    ],
)
def test_cpu_runner_supports_each_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
    workload_value: dict[str, object],
    expected_count: int,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = json.loads(
        (ROOT / "configs" / "rewrites" / config_name).read_text()
    )
    resolved = resolve_rewrite_config(config)
    adapter = get_family_adapter(resolved["family_id"])
    workload = adapter.workload_from_dict(workload_value)
    protocol = {
        "backend": "eager",
        "rounds": 1,
        "warmup_per_round": 0,
        "samples_per_round": 1,
        "iterations_per_sample": 1,
        "precondition_seconds": 0,
        "monitor_mode": "off",
        "monitor_backend": "nvml",
        "monitor_interval_seconds": 1,
        "bootstrap_resamples": 10,
        "randomization_seed": 20260717,
        "profile_execution_classes": True,
        "fingerprint_noise_floor_relative": 0.02,
        "same_class_audit": {"enabled": False},
        "equivalence": {
            "seeds": [0],
            "distributions": ["normal", "zeros"],
            "atol_by_dtype": {"fp32": 0.00001},
            "rtol_by_dtype": {"fp32": 0.00001},
            "atol": 0.00001,
            "rtol": 0.00001,
        },
    }

    result = profile_workload(
        workload,
        resolved["plans"],
        protocol,
        tmp_path / workload.group_id,
        adapter,
    )

    assert result["family_id"] == adapter.family_id
    assert result["num_valid_candidates"] == expected_count
    assert (tmp_path / workload.group_id / "result.json").is_file()
