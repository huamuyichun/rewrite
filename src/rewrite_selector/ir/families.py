from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from rewrite_selector.ir.mlp import (
    Workload,
    instantiate_candidate,
    make_baseline,
    make_input,
)
from rewrite_selector.ir.rmsnorm import (
    RMSNormWorkload,
    instantiate_rmsnorm_candidate,
    make_rmsnorm_baseline,
    make_rmsnorm_input,
)


@dataclass(frozen=True)
class FamilyAdapter:
    family_id: str
    workload_type: type[Any]
    baseline_factory: Callable[..., torch.nn.Module]
    candidate_factory: Callable[..., torch.nn.Module]
    input_factory: Callable[..., torch.Tensor]

    def workload_from_dict(self, value: dict[str, Any]) -> Any:
        return self.workload_type.from_dict(value)


FAMILY_ADAPTERS = {
    "mlp_gate_up_activation_control": FamilyAdapter(
        family_id="mlp_gate_up_activation_control",
        workload_type=Workload,
        baseline_factory=make_baseline,
        candidate_factory=instantiate_candidate,
        input_factory=make_input,
    ),
    "rmsnorm_residual_boundary": FamilyAdapter(
        family_id="rmsnorm_residual_boundary",
        workload_type=RMSNormWorkload,
        baseline_factory=make_rmsnorm_baseline,
        candidate_factory=instantiate_rmsnorm_candidate,
        input_factory=make_rmsnorm_input,
    ),
}


def get_family_adapter(family_id: str) -> FamilyAdapter:
    try:
        return FAMILY_ADAPTERS[family_id]
    except KeyError as exc:
        raise ValueError(f"unsupported rewrite family: {family_id}") from exc
