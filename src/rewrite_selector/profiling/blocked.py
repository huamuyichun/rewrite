from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from typing import Any

import torch

from rewrite_selector.evaluation.statistics import summarize_latency
from rewrite_selector.profiling.environment import gpu_snapshot, is_contaminated


def blocked_schedule(
    candidate_ids: list[str],
    rounds: int,
    seed: int,
) -> list[list[str]]:
    rng = random.Random(seed)
    schedule: list[list[str]] = []
    for _ in range(rounds):
        order = list(candidate_ids)
        rng.shuffle(order)
        schedule.append(order)
    return schedule


def measure_once(
    fn: Callable[[torch.Tensor], torch.Tensor],
    example: torch.Tensor,
    iterations: int = 1,
) -> float:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if example.device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        with torch.no_grad():
            for _ in range(iterations):
                output = fn(example)
        end.record()
        torch.cuda.synchronize()
        _ = output.detach()
        return float(start.elapsed_time(end)) / iterations
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            output = fn(example)
    _ = output.detach()
    return (time.perf_counter() - start_time) * 1000 / iterations


def run_blocked_rounds(
    callables: dict[str, Callable[[torch.Tensor], torch.Tensor]],
    example: torch.Tensor,
    rounds: int,
    warmup_per_round: int,
    samples_per_round: int,
    iterations_per_sample: int,
    randomization_seed: int,
    bootstrap_resamples: int,
    precondition_seconds: float = 0.0,
    monitor_mode: str = "async",
    monitor_backend: str = "nvml",
    monitor_interval_seconds: float = 0.25,
) -> dict[str, Any]:
    if monitor_mode not in {"off", "async"}:
        raise ValueError(f"unsupported monitor mode: {monitor_mode}")
    if monitor_interval_seconds <= 0:
        raise ValueError("monitor_interval_seconds must be positive")

    schedule = blocked_schedule(list(callables), rounds, randomization_seed)
    raw: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = [gpu_snapshot(monitor_backend)]

    deadline = time.perf_counter() + precondition_seconds
    while time.perf_counter() < deadline:
        for fn in callables.values():
            measure_once(fn, example, iterations_per_sample)
            if time.perf_counter() >= deadline:
                break

    monitor_stop = threading.Event()
    monitor_metrics = {
        "sample_count": 0,
        "snapshot_wall_seconds": 0.0,
        "thread_cpu_seconds": 0.0,
    }

    def monitor_gpu() -> None:
        cpu_started = time.thread_time()
        while not monitor_stop.wait(monitor_interval_seconds):
            wall_started = time.perf_counter()
            snapshots.append(gpu_snapshot(monitor_backend))
            monitor_metrics["sample_count"] += 1
            monitor_metrics["snapshot_wall_seconds"] += (
                time.perf_counter() - wall_started
            )
        monitor_metrics["thread_cpu_seconds"] = time.thread_time() - cpu_started

    monitor_thread: threading.Thread | None = None
    if monitor_mode == "async":
        monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
        monitor_thread.start()

    measurement_started_ns = time.time_ns()
    for round_index, order in enumerate(schedule):
        round_started_ns = time.time_ns()
        for order_index, candidate_id in enumerate(order):
            fn = callables[candidate_id]
            for _ in range(warmup_per_round):
                measure_once(fn, example, iterations_per_sample)
            for sample_index in range(samples_per_round):
                sample_started_ns = time.time_ns()
                latency_ms = measure_once(fn, example, iterations_per_sample)
                raw.append(
                    {
                        "round_index": round_index,
                        "order_index": order_index,
                        "candidate_id": candidate_id,
                        "sample_index": sample_index,
                        "iterations_per_sample": iterations_per_sample,
                        "started_ns": sample_started_ns,
                        "ended_ns": time.time_ns(),
                        "latency_ms": latency_ms,
                    }
                )
        round_rows.append(
            {
                "round_index": round_index,
                "started_ns": round_started_ns,
                "ended_ns": time.time_ns(),
                "candidate_order": order,
            }
        )
    measurement_ended_ns = time.time_ns()

    monitor_stop.set()
    if monitor_thread is not None:
        monitor_thread.join(timeout=max(2.0, monitor_interval_seconds * 2))
    snapshots.append(gpu_snapshot(monitor_backend))

    boundary_contaminated = bool(
        snapshots[0].get("foreign_processes")
        or snapshots[-1].get("foreign_processes")
    )
    contaminated_rounds = 0
    for round_row in round_rows:
        in_round = [
            snapshot
            for snapshot in snapshots
            if round_row["started_ns"]
            <= int(snapshot["timestamp_ns"])
            <= round_row["ended_ns"]
        ]
        round_row["monitor_samples"] = len(in_round)
        round_row["contaminated"] = boundary_contaminated or any(
            snapshot.get("foreign_processes") for snapshot in in_round
        )
        contaminated_rounds += int(round_row["contaminated"])

    summary: dict[str, dict[str, float | int]] = {}
    for candidate_index, candidate_id in enumerate(callables):
        values = [
            row["latency_ms"]
            for row in raw
            if row["candidate_id"] == candidate_id
        ]
        summary[candidate_id] = summarize_latency(
            values,
            resamples=bootstrap_resamples,
            seed=randomization_seed + candidate_index,
        )

    return {
        "schedule": schedule,
        "rounds": round_rows,
        "raw": raw,
        "candidate_summary": summary,
        "gpu_snapshots": snapshots,
        "contaminated": is_contaminated(snapshots),
        "contaminated_rounds": contaminated_rounds,
        "contaminated_round_ratio": (
            contaminated_rounds / len(round_rows) if round_rows else 0.0
        ),
        "measurement_started_ns": measurement_started_ns,
        "measurement_ended_ns": measurement_ended_ns,
        "monitor": {
            "mode": monitor_mode,
            "backend": monitor_backend,
            "polling_interval_seconds": monitor_interval_seconds,
            "sample_count": monitor_metrics["sample_count"],
            "snapshot_wall_seconds": monitor_metrics["snapshot_wall_seconds"],
            "thread_cpu_seconds": monitor_metrics["thread_cpu_seconds"],
            "subprocess_launches": (
                int(monitor_metrics["sample_count"]) * 2
                if monitor_mode == "async" and monitor_backend == "nvidia_smi"
                else 0
            ),
        },
    }
