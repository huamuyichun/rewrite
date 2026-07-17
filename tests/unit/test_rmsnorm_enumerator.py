import torch

from rewrite_selector.ir.rmsnorm import (
    RMSNormWorkload,
    instantiate_rmsnorm_candidate,
    make_rmsnorm_baseline,
    make_rmsnorm_input,
)
from rewrite_selector.rewrites.rmsnorm_enumerator import (
    enumerate_rmsnorm_candidates,
)


def test_rmsnorm_enumerator_is_bounded_and_deterministic() -> None:
    first = enumerate_rmsnorm_candidates()
    second = enumerate_rmsnorm_candidates()
    assert first == second
    assert 4 <= len(first["candidates"]) <= 16
    assert len({item["fx_sha256"] for item in first["candidates"]}) == len(
        first["candidates"]
    )
    assert sum(item["is_baseline"] for item in first["candidates"]) == 1


def test_rmsnorm_candidates_match_native_fp32() -> None:
    result = enumerate_rmsnorm_candidates()
    workload = RMSNormWorkload(
        "test",
        "tiny",
        "prefill",
        1,
        3,
        16,
        "fp32",
        7,
        "residual_silu",
    )
    device = torch.device("cpu")
    baseline = make_rmsnorm_baseline(workload, device, torch.float32)
    example = make_rmsnorm_input(
        workload,
        device,
        torch.float32,
        7,
        "normal",
    )
    reference = baseline(example)
    for plan in result["candidates"]:
        candidate = instantiate_rmsnorm_candidate(
            plan,
            workload,
            baseline,
            device,
            torch.float32,
        )
        assert torch.allclose(
            reference,
            candidate(example),
            atol=2e-6,
            rtol=2e-6,
        ), plan["candidate_id"]
