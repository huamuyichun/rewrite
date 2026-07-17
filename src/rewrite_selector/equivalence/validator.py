from __future__ import annotations

from typing import Any, Callable

import torch
from torch.fx import GraphModule, symbolic_trace

from rewrite_selector.ir.mlp import Workload, make_input


def check_inplace_safety(module: torch.nn.Module) -> dict[str, Any]:
    traced = symbolic_trace(module)
    violations: list[str] = []
    inplace_nodes = 0
    for node in traced.graph.nodes:
        target = str(node.target)
        if "mul_" not in target:
            continue
        inplace_nodes += 1
        source = node.args[0] if node.args else None
        if not hasattr(source, "users") or len(source.users) != 1:
            violations.append(f"{node.name}: mutated source has {len(source.users) if hasattr(source, 'users') else 'unknown'} users")
    return {
        "status": "ok" if not violations else "failed",
        "num_inplace_nodes": inplace_nodes,
        "violations": violations,
    }


def error_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    diff = (reference.detach().float() - actual.detach().float()).abs()
    denom = reference.detach().float().abs().clamp_min(1e-6)
    return {
        "max_abs_diff": float(diff.max().cpu()),
        "mean_abs_diff": float(diff.mean().cpu()),
        "max_rel_diff": float((diff / denom).max().cpu()),
    }


def validate_callable(
    reference: Callable[[torch.Tensor], torch.Tensor],
    candidate: Callable[[torch.Tensor], torch.Tensor],
    workload: Workload,
    device: torch.device,
    dtype: torch.dtype,
    seeds: list[int],
    distributions: list[str],
    atol: float,
    rtol: float,
    input_factory: Callable[..., torch.Tensor] = make_input,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    with torch.no_grad():
        for seed in seeds:
            for distribution in distributions:
                example = input_factory(workload, device, dtype, seed, distribution)
                expected = reference(example)
                actual = candidate(example)
                metrics = error_metrics(expected, actual)
                cases.append(
                    {
                        "seed": seed,
                        "distribution": distribution,
                        "allclose": bool(torch.allclose(expected.float(), actual.float(), atol=atol, rtol=rtol)),
                        **metrics,
                    }
                )
    return {
        "status": "ok" if all(case["allclose"] for case in cases) else "failed",
        "atol": atol,
        "rtol": rtol,
        "num_cases": len(cases),
        "cases": cases,
    }

