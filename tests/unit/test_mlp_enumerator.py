import torch

from rewrite_selector.equivalence.validator import validate_callable
from rewrite_selector.ir.mlp import (
    Workload,
    instantiate_candidate,
    make_baseline,
)
from rewrite_selector.rewrites.mlp_enumerator import (
    enumerate_mlp_candidates,
)


def test_bounded_enumeration_is_deterministic_and_unique() -> None:
    first = enumerate_mlp_candidates(max_depth=3, max_candidates=32)
    second = enumerate_mlp_candidates(max_depth=3, max_candidates=32)
    assert first == second
    candidates = first["candidates"]
    assert 6 <= len(candidates) <= 32
    assert len({item["candidate_id"] for item in candidates}) == len(candidates)
    assert len({item["fx_sha256"] for item in candidates}) == len(candidates)
    assert sum(item["is_baseline"] for item in candidates) == 1
    assert all(item["min_rewrite_depth"] <= 3 for item in candidates)
    assert all(rule["hypothesis"] for rule in first["rule_registry"])


def test_enumerated_candidates_are_numerically_equivalent() -> None:
    result = enumerate_mlp_candidates(max_depth=3, max_candidates=32)
    workload = Workload(
        "enumerator_test",
        "tiny",
        "prefill",
        1,
        3,
        8,
        16,
        "fp32",
        7,
    )
    device = torch.device("cpu")
    baseline = make_baseline(workload, device, torch.float32)
    for plan in result["candidates"]:
        candidate = instantiate_candidate(
            plan,
            workload,
            baseline,
            device,
            torch.float32,
        )
        validation = validate_callable(
            baseline,
            candidate,
            workload,
            device,
            torch.float32,
            seeds=[0, 1],
            distributions=["normal", "zeros", "extremes"],
            atol=1e-6,
            rtol=1e-5,
        )
        assert validation["status"] == "ok", plan["candidate_id"]
