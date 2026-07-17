#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
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

from rewrite_selector.equivalence.validator import (
    check_inplace_safety,
    validate_callable,
)
from rewrite_selector.evaluation.execution_classes import (
    analyze_fingerprint_consistency,
    build_execution_classes,
    class_by_candidate,
    class_summary_from_profile,
    representative_callables,
)
from rewrite_selector.ir.families import FamilyAdapter, get_family_adapter
from rewrite_selector.ir.mlp import dtype_from_name, set_seed
from rewrite_selector.lowering.fingerprint import (
    fingerprint_inductor_artifacts,
    high_level_fingerprint,
)
from rewrite_selector.profiling.blocked import run_blocked_rounds
from rewrite_selector.profiling.environment import environment_manifest
from rewrite_selector.rewrites.registry import resolve_rewrite_config


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
        ).strip()

    try:
        commit = run("rev-parse", "HEAD")
        status_lines = run("status", "--porcelain").splitlines()
        dirty = any(
            not line.endswith(" artifacts/registry.jsonl")
            for line in status_lines
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, True
    return {"commit": commit, "dirty": dirty}


def configure_inductor_trace(
    enabled: bool,
    debug_dir: Path | None = None,
) -> None:
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
    return (
        compiled,
        compile_prime_ms,
        fingerprint_inductor_artifacts(artifact_dir),
    )


def candidate_gate(
    baseline: torch.nn.Module,
    candidate: torch.nn.Module,
    compiled: Callable[[torch.Tensor], torch.Tensor],
    workload: Any,
    device: torch.device,
    dtype: torch.dtype,
    equivalence_config: dict[str, Any],
    input_factory: Callable[..., torch.Tensor],
) -> dict[str, Any]:
    atol = equivalence_config.get("atol_by_dtype", {}).get(
        workload.dtype,
        equivalence_config["atol"],
    )
    rtol = equivalence_config.get("rtol_by_dtype", {}).get(
        workload.dtype,
        equivalence_config["rtol"],
    )
    common = {
        "workload": workload,
        "device": device,
        "dtype": dtype,
        "seeds": list(equivalence_config["seeds"]),
        "distributions": list(equivalence_config["distributions"]),
        "atol": float(atol),
        "rtol": float(rtol),
        "input_factory": input_factory,
    }
    eager = validate_callable(baseline, candidate, **common)
    compiled_result = validate_callable(baseline, compiled, **common)
    alias = check_inplace_safety(candidate)
    return {
        "status": (
            "ok"
            if eager["status"]
            == compiled_result["status"]
            == alias["status"]
            == "ok"
            else "failed"
        ),
        "eager": eager,
        "compiled": compiled_result,
        "alias": alias,
    }


def _profile_options(protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "iterations_per_sample": int(
            protocol.get("iterations_per_sample", 1)
        ),
        "randomization_seed": int(protocol["randomization_seed"]),
        "bootstrap_resamples": int(protocol["bootstrap_resamples"]),
        "precondition_seconds": float(
            protocol.get("precondition_seconds", 0.0)
        ),
        "monitor_mode": str(protocol.get("monitor_mode", "async")),
        "monitor_backend": str(protocol.get("monitor_backend", "nvml")),
        "monitor_interval_seconds": float(
            protocol.get("monitor_interval_seconds", 0.25)
        ),
    }


def profile_workload(
    workload: Any,
    plans: list[dict[str, Any]],
    protocol: dict[str, Any],
    output_dir: Path,
    adapter: FamilyAdapter,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(workload.dtype)
    set_seed(workload.seed)
    baseline = adapter.baseline_factory(workload, device, dtype)
    example = adapter.input_factory(
        workload,
        device,
        dtype,
        workload.seed,
        "normal",
    )

    callables: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
    modules: dict[str, torch.nn.Module] = {}
    audits: dict[str, Any] = {}

    for plan in plans:
        candidate_id = str(plan["candidate_id"])
        candidate_dir = output_dir / "candidates" / candidate_id
        module = adapter.candidate_factory(
            plan,
            workload,
            baseline,
            device,
            dtype,
        )
        high_level = high_level_fingerprint(module, example)
        try:
            if protocol["backend"] == "compile":
                compiled, compile_prime_ms, lowered = compile_candidate(
                    module,
                    example,
                    candidate_dir / "inductor_trace",
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
                adapter.input_factory,
            )
            audits[candidate_id] = {
                "status": gate["status"],
                "compile_prime_ms": compile_prime_ms,
                "high_level": high_level,
                "lowered": lowered,
                "equivalence": gate,
                "rewrite_trace": plan.get("rewrite_trace", []),
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
                "rewrite_trace": plan.get("rewrite_trace", []),
            }
            write_json(candidate_dir / "audit.json", audits[candidate_id])

    if not callables:
        raise RuntimeError(f"no valid candidates for {workload.group_id}")

    execution_classes = build_execution_classes(plans, audits)
    candidate_to_class = class_by_candidate(execution_classes)
    profile_by_class = bool(protocol.get("profile_execution_classes", True))
    formal_callables = (
        representative_callables(execution_classes, callables)
        if profile_by_class
        else callables
    )
    options = _profile_options(protocol)
    profile = run_blocked_rounds(
        formal_callables,
        example,
        rounds=int(protocol["rounds"]),
        warmup_per_round=int(protocol["warmup_per_round"]),
        samples_per_round=int(protocol["samples_per_round"]),
        randomization_seed=options["randomization_seed"] + workload.seed,
        **{key: value for key, value in options.items() if key != "randomization_seed"},
    )

    diagnostic_profile: dict[str, Any] | None = None
    audit_config = protocol.get("same_class_audit", {})
    diagnostic_ids = (
        sorted(callables)
        if bool(audit_config.get("include_all_candidates", True))
        else sorted(
            candidate_id
            for execution_class in execution_classes
            if len(execution_class["candidate_ids"]) > 1
            for candidate_id in execution_class["candidate_ids"]
        )
    )
    if diagnostic_ids and bool(audit_config.get("enabled", True)):
        diagnostic_profile = run_blocked_rounds(
            {
                candidate_id: callables[candidate_id]
                for candidate_id in diagnostic_ids
            },
            example,
            rounds=int(audit_config.get("rounds", 5)),
            warmup_per_round=int(audit_config.get("warmup_per_round", 1)),
            samples_per_round=int(audit_config.get("samples_per_round", 1)),
            iterations_per_sample=options["iterations_per_sample"],
            randomization_seed=(
                options["randomization_seed"] + workload.seed + 100_000
            ),
            bootstrap_resamples=options["bootstrap_resamples"],
            precondition_seconds=float(
                audit_config.get("precondition_seconds", 0.25)
            ),
            monitor_mode="off",
            monitor_backend=options["monitor_backend"],
            monitor_interval_seconds=options["monitor_interval_seconds"],
        )

    consistency = analyze_fingerprint_consistency(
        execution_classes,
        (
            diagnostic_profile["candidate_summary"]
            if diagnostic_profile is not None
            else {}
        ),
        noise_floor_relative=float(
            protocol.get("fingerprint_noise_floor_relative", 0.02)
        ),
    )

    formal_summary = profile["candidate_summary"]
    ranked_ids = sorted(
        formal_summary,
        key=lambda item_id: float(formal_summary[item_id]["p50_ms"]),
    )
    baseline_candidate_id = next(
        str(plan["candidate_id"]) for plan in plans if plan["is_baseline"]
    )
    baseline_unit_id = (
        candidate_to_class[baseline_candidate_id]
        if profile_by_class
        else baseline_candidate_id
    )
    fastest_id, slowest_id = ranked_ids[0], ranked_ids[-1]
    high_hashes = {
        audit["high_level"]["sha256"]
        for audit in audits.values()
        if audit["status"] == "ok"
    }
    lowered_hashes = {
        audit["lowered"]["lowered_sha256"]
        for audit in audits.values()
        if audit["status"] == "ok"
        and audit["lowered"]["lowered_sha256"]
    }
    execution_hashes = {
        audit["lowered"]["execution_sha256"]
        for audit in audits.values()
        if audit["status"] == "ok"
        and audit["lowered"]["execution_sha256"]
    }

    result = {
        "family_id": adapter.family_id,
        "group_id": workload.group_id,
        "workload": workload.__dict__,
        "formal_selection_unit": (
            "execution_class" if profile_by_class else "candidate"
        ),
        "num_requested_candidates": len(plans),
        "num_valid_candidates": len(callables),
        "num_high_level_unique": len(high_hashes),
        "num_lowered_unique": len(lowered_hashes),
        "num_execution_unique": len(execution_hashes),
        "fx_retention": len(high_hashes) / len(plans),
        "lowered_retention": len(lowered_hashes) / len(high_hashes),
        "execution_retention": len(execution_hashes) / len(high_hashes),
        "execution_classes": execution_classes,
        "candidate_to_execution_class": candidate_to_class,
        "fingerprint_consistency": consistency,
        "baseline_candidate_id": baseline_candidate_id,
        "baseline_execution_class_id": candidate_to_class[
            baseline_candidate_id
        ],
        "best_unit_id": fastest_id,
        "best_p50_ms": formal_summary[fastest_id]["p50_ms"],
        "baseline_p50_ms": formal_summary[baseline_unit_id]["p50_ms"],
        "baseline_to_best_gain": (
            float(formal_summary[baseline_unit_id]["p50_ms"])
            / float(formal_summary[fastest_id]["p50_ms"])
            - 1.0
        ),
        "spread": (
            float(formal_summary[slowest_id]["p50_ms"])
            / float(formal_summary[fastest_id]["p50_ms"])
            - 1.0
        ),
        "contaminated": profile["contaminated"],
        "contaminated_round_ratio": profile["contaminated_round_ratio"],
        "candidate_audits": audits,
        "candidate_diagnostic_profile": diagnostic_profile,
        "profile": profile,
        "execution_class_summary": (
            class_summary_from_profile(execution_classes, formal_summary)
            if profile_by_class
            else []
        ),
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


def registry_entry(
    *,
    run_id: str,
    session_id: str,
    status: str,
    source: dict[str, Any],
    resolved_config: Path,
    session_dir: Path,
    cache_policy: str,
    cache_preexisting: bool,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "experiment-registry-v2",
        "record_type": "session",
        "run_id": run_id,
        "session_id": session_id,
        "status": status,
        "reason": reason,
        "source_commit": source["commit"],
        "source_dirty_at_run": source["dirty"],
        "source_state": "git",
        "config_sha256": file_sha256(resolved_config),
        "artifact_path": str(session_dir.relative_to(ROOT)),
        "environment_manifest_path": str(
            (session_dir / "environment.json").relative_to(ROOT)
        ),
        "cache_policy": cache_policy,
        "cache_preexisting": cache_preexisting,
        "eligible_for_latency_aggregation": status == "ok",
        "eligible_for_fingerprint_aggregation": status == "ok",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one independent rewrite profiling session"
    )
    parser.add_argument(
        "--rewrites",
        type=Path,
        default=ROOT / "configs/rewrites/mlp_control_v1.json",
    )
    parser.add_argument(
        "--workloads",
        type=Path,
        default=ROOT / "configs/workloads/phase1_pilot_v1.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/profiling/phase1_canary_v1.json",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--group-id", action="append", default=[])
    parser.add_argument(
        "--monitor-mode",
        choices=["off", "async"],
        default=None,
    )
    parser.add_argument(
        "--monitor-backend",
        choices=["nvml", "nvidia_smi"],
        default=None,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/phase1",
    )
    args = parser.parse_args()

    source = git_state()
    session_dir = args.output_root / args.run_id / args.session_id
    if session_dir.exists():
        raise FileExistsError(f"session already exists: {session_dir}")
    session_dir.mkdir(parents=True)
    cache_dir = session_dir / "inductor_cache"
    cache_preexisting = cache_dir.exists() and any(cache_dir.iterdir())
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)

    rewrite_config = resolve_rewrite_config(read_json(args.rewrites))
    adapter = get_family_adapter(str(rewrite_config["family_id"]))
    workload_config = read_json(args.workloads)
    workload_family = str(
        workload_config.get("family_id", adapter.family_id)
    )
    if workload_family != adapter.family_id:
        raise ValueError(
            f"workload family mismatch: {workload_family}"
        )
    protocol = read_json(args.protocol)
    if args.monitor_mode is not None:
        protocol["monitor_mode"] = args.monitor_mode
    if args.monitor_backend is not None:
        protocol["monitor_backend"] = args.monitor_backend
    selected = [
        adapter.workload_from_dict(value)
        for value in workload_config["workloads"]
        if not args.group_id or value["group_id"] in set(args.group_id)
    ]
    if not selected:
        raise ValueError("no workloads selected")

    manifest = environment_manifest()
    manifest["source"] = source
    manifest["cache"] = {
        "policy": protocol.get("cache_policy", "unspecified"),
        "directory": str(cache_dir),
        "preexisting": cache_preexisting,
    }
    write_json(session_dir / "environment.json", manifest)
    resolved_config_path = session_dir / "resolved_config.json"
    write_json(
        resolved_config_path,
        {
            "rewrite_config": rewrite_config,
            "workload_config": {
                "workloads": [workload.__dict__ for workload in selected]
            },
            "protocol": protocol,
        },
    )

    started = time.perf_counter()
    status = "ok"
    reason: str | None = None
    try:
        groups = [
            profile_workload(
                workload,
                rewrite_config["plans"],
                protocol,
                session_dir / "groups" / workload.group_id,
                adapter,
            )
            for workload in selected
        ]
        summary = {
            "status": "ok",
            "run_id": args.run_id,
            "session_id": args.session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "source": source,
            "cache": manifest["cache"],
            "groups": [
                {
                    key: group[key]
                    for key in (
                        "group_id",
                        "family_id",
                        "formal_selection_unit",
                        "num_valid_candidates",
                        "num_high_level_unique",
                        "num_lowered_unique",
                        "num_execution_unique",
                        "best_unit_id",
                        "baseline_to_best_gain",
                        "spread",
                        "contaminated",
                        "contaminated_round_ratio",
                    )
                }
                for group in groups
            ],
        }
        write_json(session_dir / "session_summary.json", summary)
    except KeyboardInterrupt:
        status = "aborted"
        reason = "keyboard interrupt"
        raise
    except Exception as exc:
        status = "failed"
        reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        entry = registry_entry(
            run_id=args.run_id,
            session_id=args.session_id,
            status=status,
            source=source,
            resolved_config=resolved_config_path,
            session_dir=session_dir,
            cache_policy=str(protocol.get("cache_policy", "unspecified")),
            cache_preexisting=cache_preexisting,
            reason=reason,
        )
        write_json(session_dir / "status.json", entry)
        append_registry(ROOT / "artifacts/registry.jsonl", entry)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
