#!/usr/bin/env python3
"""Day 5: profile executable rewrite candidates.

This script profiles the Day 4 executable candidate modules. It records raw
warmup/measurement latency, per-candidate statistics, candidate spread, and a
simple first-half vs second-half winner-flip stability check.

It intentionally does not implement heuristic selection, oracle evaluation
tables, learning models, or chitu integration. Those belong to later days.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch


DAY4_DIR = Path(__file__).resolve().parents[1] / "day4"
if str(DAY4_DIR) not in sys.path:
    sys.path.insert(0, str(DAY4_DIR))

from instantiate_candidates import (  # noqa: E402
    BlockConfig,
    dtype_from_name,
    instantiate_candidate,
    make_baseline,
    set_seed,
)


RAW_FIELDS = [
    "block_id",
    "backend_mode",
    "candidate_id",
    "is_baseline",
    "rewrite_family",
    "batch_size",
    "seq_len",
    "hidden_dim",
    "intermediate_dim",
    "dtype",
    "run_idx",
    "is_warmup",
    "latency_ms",
]

SUMMARY_FIELDS = [
    "block_id",
    "backend_mode",
    "candidate_id",
    "is_baseline",
    "rewrite_family",
    "batch_size",
    "seq_len",
    "hidden_dim",
    "intermediate_dim",
    "dtype",
    "num_measure_runs",
    "latency_p50_ms",
    "latency_mean_ms",
    "latency_std_ms",
    "latency_cv",
    "latency_p10_ms",
    "latency_p90_ms",
    "latency_rank_within_block_backend",
    "compile_prime_time_ms",
]

BLOCK_FIELDS = [
    "block_id",
    "backend_mode",
    "num_candidates",
    "baseline_candidate_id",
    "baseline_latency_p50_ms",
    "best_candidate_id",
    "best_latency_p50_ms",
    "candidate_spread_slowest_over_fastest",
    "median_latency_cv",
    "p90_latency_cv",
    "first_half_winner",
    "second_half_winner",
    "winner_flip",
]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    xs = sorted(float(v) for v in values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def measure_once(fn: Callable[[torch.Tensor], torch.Tensor], example: torch.Tensor, device: torch.device) -> float:
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        with torch.no_grad():
            out = fn(example)
        end.record()
        torch.cuda.synchronize()
        _ = out.detach()
        return float(start.elapsed_time(end))

    start_t = time.perf_counter()
    with torch.no_grad():
        out = fn(example)
    _ = out.detach()
    return (time.perf_counter() - start_t) * 1000.0


def prepare_callable(
    module: torch.nn.Module,
    example: torch.Tensor,
    backend_mode: str,
    device: torch.device,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], float]:
    compile_prime_time_ms = 0.0
    if backend_mode == "eager":
        return module, compile_prime_time_ms
    if backend_mode != "compile":
        raise ValueError(f"unsupported backend_mode: {backend_mode}")

    # Candidate modules share a small number of Python forward code objects but
    # differ in guarded attributes such as activation/multiply/split_mode. Reset
    # Dynamo before compiling each candidate so the Day 5 profiling run measures
    # per-candidate compiled graphs instead of tripping the recompile cache limit.
    torch._dynamo.reset()
    compiled = torch.compile(module, backend="inductor")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = compiled(example)
    if device.type == "cuda":
        torch.cuda.synchronize()
    _ = out.detach()
    compile_prime_time_ms = (time.perf_counter() - t0) * 1000.0
    return compiled, compile_prime_time_ms


def summarize_candidate(measure_values: list[float]) -> dict[str, float]:
    latency_mean = mean(measure_values)
    latency_std = std(measure_values)
    return {
        "num_measure_runs": len(measure_values),
        "latency_p50_ms": percentile(measure_values, 0.5),
        "latency_mean_ms": latency_mean,
        "latency_std_ms": latency_std,
        "latency_cv": latency_std / latency_mean if latency_mean > 0 else float("nan"),
        "latency_p10_ms": percentile(measure_values, 0.1),
        "latency_p90_ms": percentile(measure_values, 0.9),
    }


def half_winner(candidate_measurements: dict[str, list[float]], first_half: bool) -> str:
    scores: dict[str, float] = {}
    for candidate_id, values in candidate_measurements.items():
        midpoint = max(1, len(values) // 2)
        part = values[:midpoint] if first_half else values[midpoint:]
        if not part:
            part = values
        scores[candidate_id] = percentile(part, 0.5)
    return min(scores, key=scores.get)


def profile_one_block_backend(
    block_spec: dict[str, Any],
    plans: list[dict[str, Any]],
    backend_mode: str,
    warmup: int,
    measure: int,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cfg = BlockConfig(
        batch_size=int(block_spec["batch_size"]),
        seq_len=int(block_spec["seq_len"]),
        hidden_dim=int(block_spec["hidden_dim"]),
        intermediate_dim=int(block_spec["intermediate_dim"]),
        dtype=str(block_spec["dtype"]),
        seed=int(block_spec["seed"]),
    )
    block_id = str(block_spec["block_id"])

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(cfg.dtype)
    baseline = make_baseline(cfg, device, dtype)
    example = torch.randn(
        cfg.batch_size,
        cfg.seq_len,
        cfg.hidden_dim,
        device=device,
        dtype=dtype,
    )

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    candidate_measurements: dict[str, list[float]] = {}

    for plan in plans:
        candidate_id = plan["candidate_id"]
        module = instantiate_candidate(plan, cfg, baseline, device, dtype)
        fn, compile_prime_time_ms = prepare_callable(module, example, backend_mode, device)

        measure_values: list[float] = []
        for run_idx in range(warmup + measure):
            latency_ms = measure_once(fn, example, device)
            is_warmup = run_idx < warmup
            raw_rows.append(
                {
                    "block_id": block_id,
                    "backend_mode": backend_mode,
                    "candidate_id": candidate_id,
                    "is_baseline": plan["is_baseline"],
                    "rewrite_family": plan["rewrite_family"],
                    "batch_size": cfg.batch_size,
                    "seq_len": cfg.seq_len,
                    "hidden_dim": cfg.hidden_dim,
                    "intermediate_dim": cfg.intermediate_dim,
                    "dtype": cfg.dtype,
                    "run_idx": run_idx,
                    "is_warmup": is_warmup,
                    "latency_ms": latency_ms,
                }
            )
            if not is_warmup:
                measure_values.append(latency_ms)

        candidate_measurements[candidate_id] = measure_values
        stats = summarize_candidate(measure_values)
        summary_rows.append(
            {
                "block_id": block_id,
                "backend_mode": backend_mode,
                "candidate_id": candidate_id,
                "is_baseline": plan["is_baseline"],
                "rewrite_family": plan["rewrite_family"],
                "batch_size": cfg.batch_size,
                "seq_len": cfg.seq_len,
                "hidden_dim": cfg.hidden_dim,
                "intermediate_dim": cfg.intermediate_dim,
                "dtype": cfg.dtype,
                **stats,
                "latency_rank_within_block_backend": -1,
                "compile_prime_time_ms": compile_prime_time_ms,
            }
        )

        del module
        if backend_mode == "compile":
            del fn
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ranked = sorted(summary_rows, key=lambda row: float(row["latency_p50_ms"]))
    for rank, row in enumerate(ranked, start=1):
        row["latency_rank_within_block_backend"] = rank

    by_id = {row["candidate_id"]: row for row in summary_rows}
    baseline_id = next(plan["candidate_id"] for plan in plans if plan["is_baseline"])
    best = ranked[0]
    slowest = ranked[-1]
    cvs = [float(row["latency_cv"]) for row in summary_rows]
    first = half_winner(candidate_measurements, first_half=True)
    second = half_winner(candidate_measurements, first_half=False)
    block_row = {
        "block_id": block_id,
        "backend_mode": backend_mode,
        "num_candidates": len(plans),
        "baseline_candidate_id": baseline_id,
        "baseline_latency_p50_ms": by_id[baseline_id]["latency_p50_ms"],
        "best_candidate_id": best["candidate_id"],
        "best_latency_p50_ms": best["latency_p50_ms"],
        "candidate_spread_slowest_over_fastest": float(slowest["latency_p50_ms"]) / float(best["latency_p50_ms"]) - 1.0,
        "median_latency_cv": percentile(cvs, 0.5),
        "p90_latency_cv": percentile(cvs, 0.9),
        "first_half_winner": first,
        "second_half_winner": second,
        "winner_flip": first != second,
    }

    block_out = out_dir / "block_backend_outputs" / f"{block_id}_{backend_mode}"
    write_csv(block_out / "raw_profile.csv", raw_rows, RAW_FIELDS)
    write_csv(block_out / "candidate_summary.csv", summary_rows, SUMMARY_FIELDS)
    (block_out / "block_summary.json").write_text(json.dumps(block_row, indent=2), encoding="utf-8")

    return raw_rows, summary_rows, block_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, default=Path(__file__).resolve().parents[1] / "day3" / "candidate_plans.json")
    parser.add_argument("--block-specs", type=Path, default=Path(__file__).resolve().parents[1] / "day4" / "block_specs.json")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--backend-modes", nargs="+", choices=["eager", "compile"], default=["compile"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measure", type=int, default=30)
    args = parser.parse_args()

    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/pub/data/hjwz/.cache/torchinductor")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plans = json.loads(args.plans.read_text(encoding="utf-8"))
    block_specs = json.loads(args.block_specs.read_text(encoding="utf-8"))

    all_raw: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []

    for backend_mode in args.backend_modes:
        for block_spec in block_specs:
            raw_rows, summary_rows, block_row = profile_one_block_backend(
                block_spec=block_spec,
                plans=plans,
                backend_mode=backend_mode,
                warmup=args.warmup,
                measure=args.measure,
                out_dir=args.out_dir,
            )
            all_raw.extend(raw_rows)
            all_summary.extend(summary_rows)
            block_rows.append(block_row)

    write_csv(args.out_dir / "raw_profile.csv", all_raw, RAW_FIELDS)
    write_csv(args.out_dir / "candidate_summary.csv", all_summary, SUMMARY_FIELDS)
    write_csv(args.out_dir / "block_backend_summary.csv", block_rows, BLOCK_FIELDS)

    result = {
        "status": "ok",
        "backend_modes": args.backend_modes,
        "warmup": args.warmup,
        "measure": args.measure,
        "num_blocks": len(block_specs),
        "num_candidates_per_block": len(plans),
        "num_block_backend_cases": len(block_rows),
        "num_raw_rows": len(all_raw),
        "num_summary_rows": len(all_summary),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR", ""),
        "block_backend_rows": block_rows,
    }
    (args.out_dir / "profile_run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
