import json
from pathlib import Path

import pytest
import torch

from rewrite_selector.equivalence.validator import validate_callable
from rewrite_selector.ir.families import get_family_adapter
from rewrite_selector.rewrites.registry import (
    enumerate_from_config,
    resolve_rewrite_config,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("config_name", "workload_value"),
    [
        (
            "mlp_bounded_v1.json",
            {
                "group_id": "mlp_adapter_test",
                "model_id": "tiny",
                "phase": "prefill",
                "batch_size": 1,
                "seq_len": 2,
                "hidden_dim": 8,
                "intermediate_dim": 16,
                "dtype": "fp32",
                "seed": 7,
            },
        ),
        (
            "rmsnorm_bounded_v1.json",
            {
                "group_id": "rmsnorm_adapter_test",
                "model_id": "tiny",
                "phase": "prefill",
                "batch_size": 1,
                "seq_len": 2,
                "hidden_dim": 16,
                "dtype": "fp32",
                "seed": 7,
                "context": "residual_silu",
            },
        ),
    ],
)
def test_family_adapter_drives_equivalence(
    config_name: str,
    workload_value: dict[str, object],
) -> None:
    config = json.loads(
        (ROOT / "configs" / "rewrites" / config_name).read_text()
    )
    resolved = resolve_rewrite_config(config)
    adapter = get_family_adapter(resolved["family_id"])
    workload = adapter.workload_from_dict(workload_value)
    device = torch.device("cpu")
    baseline = adapter.baseline_factory(workload, device, torch.float32)

    for plan in (resolved["plans"][0], resolved["plans"][-1]):
        candidate = adapter.candidate_factory(
            plan,
            workload,
            baseline,
            device,
            torch.float32,
        )
        result = validate_callable(
            baseline,
            candidate,
            workload,
            device,
            torch.float32,
            seeds=[0, 1],
            distributions=["normal", "zeros"],
            atol=1e-5,
            rtol=1e-5,
            input_factory=adapter.input_factory,
        )
        assert result["status"] == "ok", plan["candidate_id"]


def test_enumerator_rejects_family_mismatch() -> None:
    config = json.loads(
        (ROOT / "configs" / "rewrites" / "mlp_bounded_v1.json").read_text()
    )
    config["family_id"] = "rmsnorm_residual_boundary"
    with pytest.raises(ValueError, match="family mismatch"):
        enumerate_from_config(config)
