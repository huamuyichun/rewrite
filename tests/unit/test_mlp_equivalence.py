import torch

from rewrite_selector.equivalence.validator import check_inplace_safety, validate_callable
from rewrite_selector.ir.mlp import Workload, instantiate_candidate, make_baseline


PLANS = [
    {"candidate_id": "separate", "gate_up_projection": "separate", "gate_up_split": "none", "activation": "manual_silu", "multiply": "out_of_place"},
    {"candidate_id": "fused", "gate_up_projection": "fused", "gate_up_split": "chunk", "activation": "f_silu", "multiply": "out_of_place"},
    {"candidate_id": "inplace", "gate_up_projection": "fused", "gate_up_split": "split", "activation": "f_silu", "multiply": "inplace"},
]


def test_control_candidates_are_equivalent_and_alias_safe() -> None:
    workload = Workload("test", "tiny", "prefill", 1, 3, 8, 16, "fp32", 1)
    device = torch.device("cpu")
    baseline = make_baseline(workload, device, torch.float32)
    for plan in PLANS:
        candidate = instantiate_candidate(plan, workload, baseline, device, torch.float32)
        result = validate_callable(
            baseline,
            candidate,
            workload,
            device,
            torch.float32,
            seeds=[0, 1],
            distributions=["normal", "uniform", "zeros", "extremes"],
            atol=1e-6,
            rtol=1e-5,
        )
        assert result["status"] == "ok"
        assert check_inplace_safety(candidate)["status"] == "ok"

