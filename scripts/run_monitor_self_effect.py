#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from rewrite_selector.ir.mlp import (
    Workload,
    dtype_from_name,
    instantiate_candidate,
    make_baseline,
    make_input,
    set_seed,
)
from rewrite_selector.profiling.blocked import run_blocked_rounds
from rewrite_selector.profiling.environment import environment_manifest


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--iterations-per-sample", type=int, default=100)
    parser.add_argument("--monitor-interval-seconds", type=float, default=1.0)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    workload = Workload(
        "monitor_self_effect",
        "synthetic_pilot",
        "prefill",
        1,
        128,
        1024,
        4096,
        "fp16",
        20260717,
    )
    plan = {
        "candidate_id": "p3_fused_chunk_silu",
        "gate_up_projection": "fused",
        "gate_up_split": "chunk",
        "activation": "f_silu",
        "multiply": "out_of_place",
    }
    device = torch.device("cuda")
    dtype = dtype_from_name(workload.dtype)
    set_seed(workload.seed)
    baseline = make_baseline(workload, device, dtype)
    candidate = instantiate_candidate(plan, workload, baseline, device, dtype)
    example = make_input(workload, device, dtype, workload.seed, "normal")
    torch._dynamo.reset()
    compiled = torch.compile(candidate, backend="inductor")
    with torch.no_grad():
        compiled(example)
    torch.cuda.synchronize()

    phases: list[dict[str, Any]] = []
    mode_order = ["off", "async", "async", "off"]
    for cycle in range(args.cycles):
        for position, mode in enumerate(mode_order):
            result = run_blocked_rounds(
                {"p3": compiled},
                example,
                rounds=args.rounds,
                warmup_per_round=0,
                samples_per_round=1,
                iterations_per_sample=args.iterations_per_sample,
                randomization_seed=20260717 + cycle * 10 + position,
                bootstrap_resamples=2000,
                precondition_seconds=3.0,
                monitor_mode=mode,
                monitor_backend="nvml",
                monitor_interval_seconds=args.monitor_interval_seconds,
            )
            phases.append(
                {
                    "cycle": cycle,
                    "position": position,
                    "mode": mode,
                    "summary": result["candidate_summary"]["p3"],
                    "monitor": result["monitor"],
                    "gpu_snapshots": result["gpu_snapshots"],
                    "contaminated": result["contaminated"],
                    "contaminated_round_ratio": result[
                        "contaminated_round_ratio"
                    ],
                }
            )

    paired: list[dict[str, float | int]] = []
    for cycle in range(args.cycles):
        cycle_phases = [phase for phase in phases if phase["cycle"] == cycle]
        off = statistics.mean(
            float(phase["summary"]["p50_ms"])
            for phase in cycle_phases
            if phase["mode"] == "off"
        )
        monitored = statistics.mean(
            float(phase["summary"]["p50_ms"])
            for phase in cycle_phases
            if phase["mode"] == "async"
        )
        paired.append(
            {
                "cycle": cycle,
                "off_p50_ms": off,
                "monitored_p50_ms": monitored,
                "relative_delta": monitored / off - 1.0,
            }
        )
    deltas = [float(row["relative_delta"]) for row in paired]
    result = {
        "schema_version": "monitor-self-effect-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment_manifest(),
        "protocol": {
            "cycles": args.cycles,
            "mode_order": mode_order,
            "rounds": args.rounds,
            "iterations_per_sample": args.iterations_per_sample,
            "monitor_backend": "nvml",
            "monitor_interval_seconds": args.monitor_interval_seconds,
            "same_compiled_callable": True,
            "same_process": True,
        },
        "phases": phases,
        "paired_cycles": paired,
        "median_relative_delta": statistics.median(deltas),
        "median_absolute_relative_delta": statistics.median(
            abs(value) for value in deltas
        ),
        "max_absolute_relative_delta": max(abs(value) for value in deltas),
        "clock_stable": all(
            len(
                {
                    snapshot.get("sm_clock_mhz")
                    for snapshot in phase["gpu_snapshots"]
                    if snapshot.get("sm_clock_mhz") is not None
                }
            )
            <= 1
            for phase in phases
        ),
        "contaminated": any(phase["contaminated"] for phase in phases),
        "elapsed_seconds": time.perf_counter(),
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
