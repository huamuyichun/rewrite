#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from rewrite_selector.equivalence.validator import check_inplace_safety, validate_callable
from rewrite_selector.ir.mlp import (
    Workload,
    dtype_from_name,
    instantiate_candidate,
    make_baseline,
    make_input,
    set_seed,
)
from rewrite_selector.lowering.fingerprint import (
    fingerprint_inductor_artifacts,
    high_level_fingerprint,
)
from rewrite_selector.profiling.blocked import run_blocked_rounds
from rewrite_selector.profiling.environment import environment_manifest


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def configure_inductor_trace(enabled: bool, debug_dir: Path | None = None) -> None:
    import torch._inductor.config as inductor_config

    inductor_config.trace.enabled = enabled
    if debug_dir is not None:
        inductor_config.trace.debug_dir = str(debug_dir)


def compile_candidate(
    module: torch.nn.Module,
    example: torch.Tensor,
    artifact_dir: Path,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], float, dict[str, Any]]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch._dynamo.reset()
    configure_inductor_trace(True, artifact_dir)
    started = time.perf_counter()
    try:
        compiled = torch.compile(module, backend="inductor")
        with torch.no_grad():
            output = compiled(example)
        if example.device.type == "cuda":
            torch.cuda.synchronize()
        _ = output.detach()
    finally:
        configure_inductor_trace(False)
    compile_prime_ms = (time.perf_counter() - started) * 1000
    return compiled, compile_prime_ms, fingerprint_inductor_artifacts(artifact_dir)


def candidate_gate(
    baseline: torch.nn.Module,
    candidate: torch.nn.Module,
    compiled: Callable[[torch.Tensor], torch.Tensor],
    workload: Workload,
    device: torch.device,
    dtype: torch.dtype,
    equivalence_config: dict[str, Any],
) -> dict[str, Any]:
    common = {
        "workload": workload,
        "device": device,
        "dtype": dtype,
        "seeds": list(equivalence_config["seeds"]),
        "distributions": list(equivalence_config["distributions"]),
        "atol": float(equivalence_config.get("atol_by_dtype", {}).get(workload.dtype, equivalence_config["atol"])),
        "rtol": float(equivalence_config.get("rtol_by_dtype", {}).get(workload.dtype, equivalence_config["rtol"])) ,
    }
    eager = validate_callable(baseline, candidate, **common)
    compiled_result = validate_callable(baseline, compiled, **common)
    alias = check_inplace_safety(candidate)
    return {
        "status": "ok"
        if eager["status"] == compiled_result["status"] == alias["status"] == "ok"
        else "failed",
        "eager": eager,
        "compiled": compiled_result,
        "alias": alias,
    }


def profile_workload(
    workload: Workload,
    plans: list[dict[str, Any]],
    protocol: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(workload.dtype)
    set_seed(workload.seed)
    baseline = make_baseline(workload, device, dtype)
    example = make_input(workload, device, dtype, workload.seed, "normal")

    callables: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
    modules: dict[str, torch.nn.Module] = {}
    audits: dict[str, Any] = {}

    for plan in plans:
        candidate_id = str(plan["candidate_id"])
        candidate_dir = output_dir / "candidates" / candidate_id
        module = instantiate_candidate(plan, workload, baseline, device, dtype)
        high_level = high_level_fingerprint(module, example)
        try:
            if protocol["backend"] == "compile":
                compiled, compile_prime_ms, lowered = compile_candidate(
                    module, example, candidate_dir / "inductor_trace"
                )
            else:
                compiled, compile_prime_ms = module, 0.0
                lowered = {
                    "artifact_files": [],
                    "lowered_sha256": None,
                    "generated_code_sha256": None,
                    "execution_sha256": None,
                    "execution_records": [],
                }
            gate = candidate_gate(
                baseline,
                module,
                compiled,
                workload,
                device,
                dtype,
                protocol["equivalence"],
            )
            audits[candidate_id] = {
                "status": gate["status"],
                "compile_prime_ms": compile_prime_ms,
                "high_level": high_level,
                "lowered": lowered,
                "equivalence": gate,
            }
            write_json(candidate_dir / "audit.json", audits[candidate_id])
            if gate["status"] == "ok":
                callables[candidate_id] = compiled
                modules[candidate_id] = module
        except Exception as exc:
            audits[candidate_id] = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "high_level": high_level,
            }
            write_json(candidate_dir / "audit.json", audits[candidate_id])

    if not callables:
        raise RuntimeError(f"no valid candidates for {workload.group_id}")

    profile = run_blocked_rounds(
        callables,
        example,
        rounds=int(protocol["rounds"]),
        warmup_per_round=int(protocol["warmup_per_round"]),
        samples_per_round=int(protocol["samples_per_round"]),
        iterations_per_sample=int(protocol.get("iterations_per_sample", 1)),
        randomization_seed=int(protocol["randomization_seed"]) + workload.seed,
        bootstrap_resamples=int(protocol["bootstrap_resamples"]),
        precondition_seconds=float(protocol.get("precondition_seconds", 0.0)),
        monitor_interval_seconds=float(protocol.get("monitor_interval_seconds", 0.25)),
    )
    summaries = profile["candidate_summary"]
    ranked = sorted(summaries, key=lambda candidate_id: float(summaries[candidate_id]["p50_ms"]))
    baseline_id = next(plan["candidate_id"] for plan in plans if plan["is_baseline"])
    fastest = ranked[0]
    slowest = ranked[-1]
    high_unique = len({audit["high_level"]["sha256"] for audit in audits.values() if audit["status"] == "ok"})
    lowered_hashes = {
        audit["lowered"]["lowered_sha256"]
        for audit in audits.values()
        if audit["status"] == "ok" and audit["lowered"]["lowered_sha256"]
    }
    execution_hashes = {
        audit["lowered"]["execution_sha256"]
        for audit in audits.values()
        if audit["status"] == "ok" and audit["lowered"]["execution_sha256"]
    }
    result = {
        "group_id": workload.group_id,
        "workload": workload.__dict__,
        "num_requested_candidates": len(plans),
        "num_valid_candidates": len(callables),
        "num_high_level_unique": high_unique,
        "num_lowered_unique": len(lowered_hashes),
        "num_execution_unique": len(execution_hashes),
        "lowered_fingerprint_coverage": len(lowered_hashes) / len(callables),
        "execution_fingerprint_coverage": len(execution_hashes) / len(callables),
        "baseline_candidate_id": baseline_id,
        "best_candidate_id": fastest,
        "best_p50_ms": summaries[fastest]["p50_ms"],
        "baseline_p50_ms": summaries.get(baseline_id, {}).get("p50_ms"),
        "spread": float(summaries[slowest]["p50_ms"]) / float(summaries[fastest]["p50_ms"]) - 1,
        "contaminated": profile["contaminated"],
        "candidate_audits": audits,
        "profile": profile,
    }
    write_json(output_dir / "result.json", result)

    del callables, modules, baseline, example
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def append_registry(registry: Path, entry: dict[str, Any]) -> None:
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one independent Phase 1 profiling session")
    parser.add_argument("--rewrites", type=Path, default=ROOT / "configs/rewrites/mlp_control_v1.json")
    parser.add_argument("--workloads", type=Path, default=ROOT / "configs/workloads/phase1_pilot_v1.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/profiling/phase1_canary_v1.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--group-id", action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/phase1")
    args = parser.parse_args()

    session_dir = args.output_root / args.run_id / args.session_id
    if session_dir.exists():
        raise FileExistsError(f"session already exists: {session_dir}")
    session_dir.mkdir(parents=True)
    cache_dir = session_dir / "inductor_cache"
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))

    rewrite_config = read_json(args.rewrites)
    workload_config = read_json(args.workloads)
    protocol = read_json(args.protocol)
    selected = [
        Workload.from_dict(value)
        for value in workload_config["workloads"]
        if not args.group_id or value["group_id"] in set(args.group_id)
    ]
    if not selected:
        raise ValueError("no workloads selected")

    manifest = environment_manifest()
    write_json(session_dir / "environment.json", manifest)
    write_json(
        session_dir / "resolved_config.json",
        {
            "rewrite_config": rewrite_config,
            "workload_config": {"workloads": [workload.__dict__ for workload in selected]},
            "protocol": protocol,
        },
    )

    started = time.perf_counter()
    groups = [
        profile_workload(workload, rewrite_config["plans"], protocol, session_dir / "groups" / workload.group_id)
        for workload in selected
    ]
    summary = {
        "status": "ok",
        "run_id": args.run_id,
        "session_id": args.session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "groups": [
            {
                key: group[key]
                for key in (
                    "group_id",
                    "num_valid_candidates",
                    "num_high_level_unique",
                    "num_lowered_unique",
                    "num_execution_unique",
                    "best_candidate_id",
                    "spread",
                    "contaminated",
                )
            }
            for group in groups
        ],
    }
    write_json(session_dir / "session_summary.json", summary)
    append_registry(ROOT / "artifacts/registry.jsonl", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

