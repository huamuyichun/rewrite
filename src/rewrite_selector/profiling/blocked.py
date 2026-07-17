from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from typing import Any

import torch

from rewrite_selector.evaluation.statistics import summarize_latency
from rewrite_selector.profiling.environment import gpu_snapshot, is_contaminated


def blocked_schedule(candidate_ids: list[str], rounds: int, seed: int) -> list[list[str]]:
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
    monitor_interval_seconds: float = 0.25,
) -> dict[str, Any]:
    schedule = blocked_schedule(list(callables), rounds, randomization_seed)
    raw: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = [gpu_snapshot()]

    deadline = time.perf_counter() + precondition_seconds
    while time.perf_counter() < deadline:
        for fn in callables.values():
            measure_once(fn, example, iterations_per_sample)
            if time.perf_counter() >= deadline:
                break
    monitor_stop = threading.Event()

    def monitor_gpu() -> None:
        while not monitor_stop.wait(monitor_interval_seconds):
            snapshots.append(gpu_snapshot())

    monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
    monitor_thread.start()

    for round_index, order in enumerate(schedule):
        for order_index, candidate_id in enumerate(order):
            fn = callables[candidate_id]
            for _ in range(warmup_per_round):
                measure_once(fn, example, iterations_per_sample)
            for sample_index in range(samples_per_round):
                raw.append(
                    {
                        "round_index": round_index,
                        "order_index": order_index,
                        "candidate_id": candidate_id,
                        "sample_index": sample_index,
                        "iterations_per_sample": iterations_per_sample,
                        "latency_ms": measure_once(fn, example, iterations_per_sample),
                    }
                )

    monitor_stop.set()
    monitor_thread.join(timeout=2.0)
    snapshots.append(gpu_snapshot())

    summary: dict[str, dict[str, float | int]] = {}
    for candidate_index, candidate_id in enumerate(callables):
        values = [row["latency_ms"] for row in raw if row["candidate_id"] == candidate_id]
        summary[candidate_id] = summarize_latency(
            values,
            resamples=bootstrap_resamples,
            seed=randomization_seed + candidate_index,
        )
    return {
        "schedule": schedule,
        "raw": raw,
        "candidate_summary": summary,
        "gpu_snapshots": snapshots,
        "contaminated": is_contaminated(snapshots),
    }

