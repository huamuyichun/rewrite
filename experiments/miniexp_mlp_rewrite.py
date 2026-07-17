#!/usr/bin/env python3
"""Minimal profiling loop for graph-level rewrite plan selection.

This script intentionally keeps the first experiment small:
- Transformer-style MLP block only.
- One rewrite family: fusion-related gate/up projection variants.
- No learned model and no serving integration.
- Dependencies: torch, numpy, and Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import random
import statistics
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


RAW_FIELDS = [
    "graph_id",
    "shape_id",
    "block_type",
    "batch_size",
    "seq_len",
    "hidden_dim",
    "intermediate_dim",
    "dtype",
    "candidate_id",
    "is_baseline",
    "rewrite_family",
    "plan_desc",
    "run_idx",
    "is_warmup",
    "latency_ms",
]

SUMMARY_FIELDS = [
    "graph_id",
    "shape_id",
    "block_type",
    "batch_size",
    "seq_len",
    "hidden_dim",
    "intermediate_dim",
    "dtype",
    "candidate_id",
    "is_baseline",
    "rewrite_family",
    "plan_desc",
    "num_measure_runs",
    "latency_p50_ms",
    "latency_mean_ms",
    "latency_std_ms",
    "latency_cv",
    "latency_rank_within_graph",
]

SELECTION_FIELDS = [
    "graph_id",
    "shape_id",
    "seq_len",
    "hidden_dim",
    "intermediate_dim",
    "selector_name",
    "num_candidates",
    "baseline_candidate_id",
    "baseline_latency_ms",
    "oracle_candidate_id",
    "oracle_latency_ms",
    "selected_candidate_id",
    "selected_latency_ms",
    "speedup_vs_baseline",
    "regret_vs_oracle",
    "decision_time_p50_ms",
    "decision_time_p95_ms",
    "win_over_2pct",
]


@dataclass(frozen=True)
class GraphSpec:
    graph_id: str
    shape_id: str
    block_type: str
    batch_size: int
    seq_len: int
    hidden_dim: int
    intermediate_dim: int
    dtype_name: str


@dataclass(frozen=True)
class CandidatePlan:
    candidate_id: str
    is_baseline: bool
    rewrite_family: str
    plan_desc: str
    fused_gate_up: bool
    split_mode: str
    activation_mode: str
    inplace_mul: bool


def percentile(values: Sequence[float], q: float) -> float:
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


def geomean(values: Sequence[float]) -> float:
    positive = [float(v) for v in values if float(v) > 0.0]
    if not positive:
        return float("nan")
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def fmt(x: float, ndigits: int = 4) -> str:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "nan"
    return f"{x:.{ndigits}f}"


def align_to(value: int, multiple: int = 256) -> int:
    return int(math.ceil(value / multiple) * multiple)


def make_graph_specs(
    seq_lens: Sequence[int],
    graphs_per_shape: int,
    dtype_name: str,
    seed: int,
) -> List[GraphSpec]:
    rng = random.Random(seed)
    specs: List[GraphSpec] = []

    # Shape-aware sizes keep the full 60-graph run practical on one A100 while
    # still preserving MLP-like matrix sizes and the required sequence lengths.
    hidden_choices_by_seq = {
        128: [512, 768, 1024, 1280, 1536, 2048],
        512: [512, 768, 1024, 1280, 1536],
        2048: [384, 512, 768, 1024, 1280],
    }
    default_hidden_choices = [512, 768, 1024, 1280]
    multipliers = [2.0, 2.5, 3.0, 4.0]

    for shape_idx, seq_len in enumerate(seq_lens):
        hidden_choices = hidden_choices_by_seq.get(seq_len, default_hidden_choices)
        for local_idx in range(graphs_per_shape):
            hidden = hidden_choices[(local_idx + shape_idx) % len(hidden_choices)]
            multiplier = multipliers[(local_idx * 3 + shape_idx) % len(multipliers)]
            # Add mild variation so instances are not just four repeated shapes.
            if rng.random() < 0.35:
                multiplier = rng.choice(multipliers)
            intermediate = align_to(int(hidden * multiplier), 256)
            graph_id = f"g{shape_idx:02d}_{local_idx:03d}"
            specs.append(
                GraphSpec(
                    graph_id=graph_id,
                    shape_id=f"s{shape_idx}_seq{seq_len}",
                    block_type="transformer_mlp_swiglu",
                    batch_size=1,
                    seq_len=seq_len,
                    hidden_dim=hidden,
                    intermediate_dim=intermediate,
                    dtype_name=dtype_name,
                )
            )
    return specs


def candidate_plans() -> List[CandidatePlan]:
    return [
        CandidatePlan(
            candidate_id="baseline_separate_silu",
            is_baseline=True,
            rewrite_family="fusion_related_gate_up_projection",
            plan_desc="default: separate gate_proj and up_proj, F.silu(gate) * up",
            fused_gate_up=False,
            split_mode="none",
            activation_mode="silu",
            inplace_mul=False,
        ),
        CandidatePlan(
            candidate_id="separate_manual_silu",
            is_baseline=False,
            rewrite_family="fusion_related_gate_up_projection",
            plan_desc="separate gate/up projections, manual gate * sigmoid(gate) * up",
            fused_gate_up=False,
            split_mode="none",
            activation_mode="manual_silu",
            inplace_mul=False,
        ),
        CandidatePlan(
            candidate_id="separate_inplace_silu",
            is_baseline=False,
            rewrite_family="fusion_related_gate_up_projection",
            plan_desc="separate gate/up projections, F.silu(gate) then inplace multiply",
            fused_gate_up=False,
            split_mode="none",
            activation_mode="silu",
            inplace_mul=True,
        ),
        CandidatePlan(
            candidate_id="fused_chunk_silu",
            is_baseline=False,
            rewrite_family="fusion_related_gate_up_projection",
            plan_desc="fused gate/up projection, chunk split, F.silu(gate) * up",
            fused_gate_up=True,
            split_mode="chunk",
            activation_mode="silu",
            inplace_mul=False,
        ),
        CandidatePlan(
            candidate_id="fused_split_inplace_silu",
            is_baseline=False,
            rewrite_family="fusion_related_gate_up_projection",
            plan_desc="fused gate/up projection, torch.split, F.silu(gate) then inplace multiply",
            fused_gate_up=True,
            split_mode="split",
            activation_mode="silu",
            inplace_mul=True,
        ),
        CandidatePlan(
            candidate_id="fused_chunk_manual_silu",
            is_baseline=False,
            rewrite_family="fusion_related_gate_up_projection",
            plan_desc="fused gate/up projection, chunk split, manual gate * sigmoid(gate) * up",
            fused_gate_up=True,
            split_mode="chunk",
            activation_mode="manual_silu",
            inplace_mul=False,
        ),
    ]


def dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def make_tensors(spec: GraphSpec, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    h = spec.hidden_dim
    i = spec.intermediate_dim
    scale = 1.0 / math.sqrt(h)
    x = torch.randn(
        (spec.batch_size, spec.seq_len, h),
        device=device,
        dtype=dtype,
    )
    w_gate = torch.randn((i, h), device=device, dtype=dtype) * scale
    w_up = torch.randn((i, h), device=device, dtype=dtype) * scale
    w_down = torch.randn((h, i), device=device, dtype=dtype) * (1.0 / math.sqrt(i))
    w_fused = torch.cat([w_gate, w_up], dim=0).contiguous()
    return {
        "x": x,
        "w_gate": w_gate,
        "w_up": w_up,
        "w_down": w_down,
        "w_fused": w_fused,
    }


def run_candidate(plan: CandidatePlan, tensors: Dict[str, torch.Tensor], intermediate_dim: int) -> torch.Tensor:
    x = tensors["x"]
    if plan.fused_gate_up:
        gate_up = F.linear(x, tensors["w_fused"])
        if plan.split_mode == "chunk":
            gate, up = gate_up.chunk(2, dim=-1)
        elif plan.split_mode == "split":
            gate, up = torch.split(gate_up, intermediate_dim, dim=-1)
        else:
            raise ValueError(f"bad split mode: {plan.split_mode}")
    else:
        gate = F.linear(x, tensors["w_gate"])
        up = F.linear(x, tensors["w_up"])

    if plan.activation_mode == "manual_silu":
        hidden = gate * torch.sigmoid(gate)
        hidden = hidden * up
    elif plan.activation_mode == "silu":
        hidden = F.silu(gate)
        if plan.inplace_mul:
            hidden.mul_(up)
        else:
            hidden = hidden * up
    else:
        raise ValueError(f"bad activation mode: {plan.activation_mode}")

    return F.linear(hidden, tensors["w_down"])


def measure_candidate(
    spec: GraphSpec,
    plan: CandidatePlan,
    tensors: Dict[str, torch.Tensor],
    warmup: int,
    measure: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    total = warmup + measure
    with torch.inference_mode():
        for run_idx in range(total):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            out = run_candidate(plan, tensors, spec.intermediate_dim)
            end.record()
            torch.cuda.synchronize()
            # Keep a live reference until after synchronization.
            _ = out
            latency_ms = float(start.elapsed_time(end))
            rows.append(
                {
                    "graph_id": spec.graph_id,
                    "shape_id": spec.shape_id,
                    "block_type": spec.block_type,
                    "batch_size": spec.batch_size,
                    "seq_len": spec.seq_len,
                    "hidden_dim": spec.hidden_dim,
                    "intermediate_dim": spec.intermediate_dim,
                    "dtype": spec.dtype_name,
                    "candidate_id": plan.candidate_id,
                    "is_baseline": int(plan.is_baseline),
                    "rewrite_family": plan.rewrite_family,
                    "plan_desc": plan.plan_desc,
                    "run_idx": run_idx,
                    "is_warmup": int(run_idx < warmup),
                    "latency_ms": latency_ms,
                }
            )
    return rows


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_candidates(raw_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    metadata: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in raw_rows:
        key = (str(row["graph_id"]), str(row["candidate_id"]))
        metadata[key] = row
        if int(row["is_warmup"]) == 0:
            grouped.setdefault(key, []).append(row)

    summaries: List[Dict[str, object]] = []
    for key, rows in grouped.items():
        latencies = [float(r["latency_ms"]) for r in rows]
        meta = metadata[key]
        mean = statistics.fmean(latencies)
        std = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
        summary = {
            "graph_id": meta["graph_id"],
            "shape_id": meta["shape_id"],
            "block_type": meta["block_type"],
            "batch_size": meta["batch_size"],
            "seq_len": meta["seq_len"],
            "hidden_dim": meta["hidden_dim"],
            "intermediate_dim": meta["intermediate_dim"],
            "dtype": meta["dtype"],
            "candidate_id": meta["candidate_id"],
            "is_baseline": meta["is_baseline"],
            "rewrite_family": meta["rewrite_family"],
            "plan_desc": meta["plan_desc"],
            "num_measure_runs": len(latencies),
            "latency_p50_ms": percentile(latencies, 0.50),
            "latency_mean_ms": mean,
            "latency_std_ms": std,
            "latency_cv": std / mean if mean > 0.0 else float("nan"),
            "latency_rank_within_graph": -1,
        }
        summaries.append(summary)

    by_graph: Dict[str, List[Dict[str, object]]] = {}
    for row in summaries:
        by_graph.setdefault(str(row["graph_id"]), []).append(row)
    for rows in by_graph.values():
        rows.sort(key=lambda r: float(r["latency_p50_ms"]))
        for rank, row in enumerate(rows, start=1):
            row["latency_rank_within_graph"] = rank

    summaries.sort(key=lambda r: (str(r["graph_id"]), str(r["candidate_id"])))
    return summaries


def plan_features(spec: GraphSpec, plan: CandidatePlan) -> Dict[str, float]:
    h = spec.hidden_dim
    i = spec.intermediate_dim
    tokens = spec.batch_size * spec.seq_len
    dtype_bytes = 2.0
    input_projection_ops = 1.0 if plan.fused_gate_up else 2.0
    linear_ops = input_projection_ops + 1.0
    elementwise_ops = 3.0 if plan.activation_mode == "manual_silu" else 2.0
    output_allocs = 2.0
    if plan.inplace_mul:
        output_allocs -= 1.0
    read_write_bytes = dtype_bytes * tokens * (
        h
        + 2.0 * i
        + i
        + h
    )
    weight_bytes = dtype_bytes * (2.0 * h * i + h * i)
    launch_proxy = linear_ops + elementwise_ops
    return {
        "linear_ops": linear_ops,
        "elementwise_ops": elementwise_ops,
        "output_allocs": output_allocs,
        "read_write_bytes": read_write_bytes,
        "weight_bytes": weight_bytes,
        "launch_proxy": launch_proxy,
        "uses_fused_gate_up": 1.0 if plan.fused_gate_up else 0.0,
        "uses_manual_silu": 1.0 if plan.activation_mode == "manual_silu" else 0.0,
        "uses_inplace_mul": 1.0 if plan.inplace_mul else 0.0,
    }


def heuristic_score(spec: GraphSpec, plan: CandidatePlan) -> Tuple[float, float, float, float, str]:
    feat = plan_features(spec, plan)
    tokens = spec.batch_size * spec.seq_len
    # Lower score is better. This is intentionally simple and explainable:
    # short sequences are launch-overhead sensitive, while long sequences are
    # GEMM-shape sensitive and do not always benefit from concatenating gate/up.
    fusion_term = -7.0 if tokens <= 256 else 18.0
    inplace_term = -0.8 if (plan.inplace_mul and not plan.fused_gate_up) else 0.0
    split_penalty = 0.8 if plan.split_mode == "split" else 0.0
    score = (
        feat["linear_ops"] * 10.0
        + feat["launch_proxy"] * 2.0
        + feat["output_allocs"] * 0.5
        + feat["uses_manual_silu"] * 4.0
        + feat["uses_fused_gate_up"] * fusion_term
        + inplace_term
        + split_penalty
    )
    return (
        score,
        feat["linear_ops"],
        feat["elementwise_ops"],
        feat["output_allocs"],
        plan.candidate_id,
    )


def choose_baseline(_spec: GraphSpec, plans: Sequence[CandidatePlan], _latency_by_id: Dict[str, float]) -> str:
    for plan in plans:
        if plan.is_baseline:
            return plan.candidate_id
    raise RuntimeError("missing baseline plan")


def choose_heuristic(spec: GraphSpec, plans: Sequence[CandidatePlan], _latency_by_id: Dict[str, float]) -> str:
    return min(plans, key=lambda p: heuristic_score(spec, p)).candidate_id


def choose_oracle(_spec: GraphSpec, plans: Sequence[CandidatePlan], latency_by_id: Dict[str, float]) -> str:
    return min((p.candidate_id for p in plans), key=lambda cid: latency_by_id[cid])


def measure_selector_time(
    selector_name: str,
    spec: GraphSpec,
    plans: Sequence[CandidatePlan],
    latency_by_id: Dict[str, float],
    repeats: int,
) -> Tuple[str, float, float]:
    if selector_name == "Baseline":
        fn = choose_baseline
    elif selector_name == "Heuristic":
        fn = choose_heuristic
    elif selector_name == "Oracle":
        fn = choose_oracle
    else:
        raise ValueError(selector_name)

    selected = fn(spec, plans, latency_by_id)
    times_ms: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        selected = fn(spec, plans, latency_by_id)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    return selected, percentile(times_ms, 0.50), percentile(times_ms, 0.95)


def compute_selection_results(
    specs: Sequence[GraphSpec],
    plans: Sequence[CandidatePlan],
    summaries: Sequence[Dict[str, object]],
    decision_repeats: int,
) -> List[Dict[str, object]]:
    summary_by_graph: Dict[str, List[Dict[str, object]]] = {}
    for row in summaries:
        summary_by_graph.setdefault(str(row["graph_id"]), []).append(row)

    spec_by_id = {s.graph_id: s for s in specs}
    selection_rows: List[Dict[str, object]] = []
    for graph_id, rows in sorted(summary_by_graph.items()):
        spec = spec_by_id[graph_id]
        latency_by_id = {
            str(row["candidate_id"]): float(row["latency_p50_ms"])
            for row in rows
        }
        baseline_id = choose_baseline(spec, plans, latency_by_id)
        oracle_id = choose_oracle(spec, plans, latency_by_id)
        baseline_latency = latency_by_id[baseline_id]
        oracle_latency = latency_by_id[oracle_id]

        for selector_name in ["Baseline", "Heuristic", "Oracle"]:
            selected_id, p50_decision, p95_decision = measure_selector_time(
                selector_name,
                spec,
                plans,
                latency_by_id,
                decision_repeats,
            )
            selected_latency = latency_by_id[selected_id]
            speedup = baseline_latency / selected_latency
            regret = selected_latency / oracle_latency - 1.0
            win = int(selected_latency <= 0.98 * baseline_latency)
            selection_rows.append(
                {
                    "graph_id": graph_id,
                    "shape_id": spec.shape_id,
                    "seq_len": spec.seq_len,
                    "hidden_dim": spec.hidden_dim,
                    "intermediate_dim": spec.intermediate_dim,
                    "selector_name": selector_name,
                    "num_candidates": len(plans),
                    "baseline_candidate_id": baseline_id,
                    "baseline_latency_ms": baseline_latency,
                    "oracle_candidate_id": oracle_id,
                    "oracle_latency_ms": oracle_latency,
                    "selected_candidate_id": selected_id,
                    "selected_latency_ms": selected_latency,
                    "speedup_vs_baseline": speedup,
                    "regret_vs_oracle": regret,
                    "decision_time_p50_ms": p50_decision,
                    "decision_time_p95_ms": p95_decision,
                    "win_over_2pct": win,
                }
            )
    return selection_rows


def aggregate_selector(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_selector: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_selector.setdefault(str(row["selector_name"]), []).append(row)
    out: List[Dict[str, object]] = []
    for selector in ["Baseline", "Heuristic", "Oracle"]:
        rs = by_selector.get(selector, [])
        if not rs:
            continue
        speedups = [float(r["speedup_vs_baseline"]) for r in rs]
        regrets = [float(r["regret_vs_oracle"]) for r in rs]
        wins = [float(r["win_over_2pct"]) for r in rs]
        p50_dec = [float(r["decision_time_p50_ms"]) for r in rs]
        p95_dec = [float(r["decision_time_p95_ms"]) for r in rs]
        out.append(
            {
                "Selector": selector,
                "Median Speedup": percentile(speedups, 0.50),
                "Geomean Speedup": geomean(speedups),
                "Median Regret": percentile(regrets, 0.50),
                "P90 Regret": percentile(regrets, 0.90),
                "Win Rate": statistics.fmean(wins),
                "P50 Decision Time (ms)": percentile(p50_dec, 0.50),
                "P95 Decision Time (ms)": percentile(p95_dec, 0.50),
            }
        )
    return out


def aggregate_by_shape(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["shape_id"]), str(row["selector_name"])), []).append(row)
    out: List[Dict[str, object]] = []
    for (shape, selector), rs in sorted(groups.items()):
        speedups = [float(r["speedup_vs_baseline"]) for r in rs]
        regrets = [float(r["regret_vs_oracle"]) for r in rs]
        wins = [float(r["win_over_2pct"]) for r in rs]
        p50_dec = [float(r["decision_time_p50_ms"]) for r in rs]
        out.append(
            {
                "Shape": shape,
                "Selector": selector,
                "Median Speedup": percentile(speedups, 0.50),
                "Median Regret": percentile(regrets, 0.50),
                "Win Rate": statistics.fmean(wins),
                "P50 Decision Time (ms)": percentile(p50_dec, 0.50),
            }
        )
    return out


def png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        raw.extend(pixels[y * stride : (y + 1) * stride])
    data = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            png_chunk(b"IDAT", zlib.compress(bytes(raw), level=9)),
            png_chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(data)


def set_pixel(pixels: bytearray, width: int, height: int, x: int, y: int, color: Tuple[int, int, int]) -> None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    idx = (y * width + x) * 3
    pixels[idx : idx + 3] = bytes(color)


def draw_rect(
    pixels: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Tuple[int, int, int],
) -> None:
    x0, x1 = sorted((max(0, x0), min(width - 1, x1)))
    y0, y1 = sorted((max(0, y0), min(height - 1, y1)))
    for y in range(y0, y1 + 1):
        row = (y * width + x0) * 3
        for _x in range(x0, x1 + 1):
            pixels[row : row + 3] = bytes(color)
            row += 3


def draw_histogram_png(path: Path, values: Sequence[float], bins: int, color: Tuple[int, int, int]) -> None:
    width, height = 900, 520
    margin_l, margin_r, margin_t, margin_b = 70, 40, 35, 70
    pixels = bytearray([255, 255, 255] * width * height)
    if not values:
        write_png(path, width, height, pixels)
        return

    vals = [float(v) for v in values if not math.isnan(float(v)) and not math.isinf(float(v))]
    lo, hi = min(vals), max(vals)
    if math.isclose(lo, hi):
        lo -= 0.5
        hi += 0.5
    pad = (hi - lo) * 0.05
    lo -= pad
    hi += pad
    counts = [0 for _ in range(bins)]
    for v in vals:
        idx = int((v - lo) / (hi - lo) * bins)
        idx = max(0, min(bins - 1, idx))
        counts[idx] += 1
    max_count = max(max(counts), 1)

    axis_color = (35, 35, 35)
    grid_color = (225, 225, 225)
    plot_x0 = margin_l
    plot_x1 = width - margin_r
    plot_y0 = margin_t
    plot_y1 = height - margin_b

    for frac in [0.25, 0.5, 0.75]:
        y = int(plot_y1 - (plot_y1 - plot_y0) * frac)
        draw_rect(pixels, width, height, plot_x0, y, plot_x1, y, grid_color)
    draw_rect(pixels, width, height, plot_x0, plot_y0, plot_x0, plot_y1, axis_color)
    draw_rect(pixels, width, height, plot_x0, plot_y1, plot_x1, plot_y1, axis_color)

    bar_area = plot_x1 - plot_x0
    gap = 3
    bar_w = max(2, int(bar_area / bins) - gap)
    for idx, count in enumerate(counts):
        x0 = plot_x0 + int(idx * bar_area / bins) + gap // 2
        x1 = min(plot_x1, x0 + bar_w)
        bar_h = int((plot_y1 - plot_y0) * count / max_count)
        y0 = plot_y1 - bar_h
        draw_rect(pixels, width, height, x0, y0, x1, plot_y1 - 1, color)

    # Reference line at 1.0 is useful for both speedup and regret+1 style plots.
    if lo <= 1.0 <= hi:
        x = plot_x0 + int((1.0 - lo) / (hi - lo) * (plot_x1 - plot_x0))
        draw_rect(pixels, width, height, x, plot_y0, x + 2, plot_y1, (190, 50, 50))

    write_png(path, width, height, pixels)


def markdown_table(rows: Sequence[Dict[str, object]], columns: Sequence[str], precision: int = 4) -> str:
    if not rows:
        return "_No rows._\n"
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        cells = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                cells.append(fmt(val, precision))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def candidate_diversity(summaries: Sequence[Dict[str, object]]) -> Dict[str, float]:
    by_graph: Dict[str, List[float]] = {}
    cvs: List[float] = []
    for row in summaries:
        by_graph.setdefault(str(row["graph_id"]), []).append(float(row["latency_p50_ms"]))
        cvs.append(float(row["latency_cv"]))
    spreads = []
    has_2pct = []
    for vals in by_graph.values():
        best = min(vals)
        worst = max(vals)
        spread = worst / best - 1.0 if best > 0 else float("nan")
        spreads.append(spread)
        has_2pct.append(float(spread >= 0.02))
    return {
        "median_candidate_spread": percentile(spreads, 0.50),
        "p90_candidate_spread": percentile(spreads, 0.90),
        "graphs_with_over_2pct_spread": statistics.fmean(has_2pct) if has_2pct else float("nan"),
        "median_latency_cv": percentile(cvs, 0.50),
        "p90_latency_cv": percentile(cvs, 0.90),
    }


def write_report(
    path: Path,
    args: argparse.Namespace,
    specs: Sequence[GraphSpec],
    plans: Sequence[CandidatePlan],
    summaries: Sequence[Dict[str, object]],
    selection_rows: Sequence[Dict[str, object]],
) -> None:
    selector_rows = aggregate_selector(selection_rows)
    shape_rows = aggregate_by_shape(selection_rows)
    diversity = candidate_diversity(summaries)
    heuristic = [r for r in selector_rows if r["Selector"] == "Heuristic"]
    heuristic_row = heuristic[0] if heuristic else None

    if heuristic_row:
        pass_basic = (
            heuristic_row["Median Speedup"] >= 1.03
            and heuristic_row["Median Regret"] <= 0.10
            and heuristic_row["Win Rate"] >= 0.55
            and heuristic_row["P50 Decision Time (ms)"] <= 10.0
        )
        pass_strong = (
            heuristic_row["Median Speedup"] >= 1.05
            and heuristic_row["Median Regret"] <= 0.07
            and heuristic_row["Win Rate"] >= 0.60
            and heuristic_row["P50 Decision Time (ms)"] <= 5.0
        )
    else:
        pass_basic = False
        pass_strong = False

    seq_lens = sorted({s.seq_len for s in specs})
    hidden_dims = sorted({s.hidden_dim for s in specs})
    inter_dims = sorted({s.intermediate_dim for s in specs})
    device_name = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"

    lines = []
    lines.append("# Minimal Rewrite Plan Selection Experiment\n")
    lines.append("## 1. 实验设定\n")
    lines.append(f"- Block 类型：Transformer MLP / SwiGLU-like block\n")
    lines.append(f"- Rewrite family：fusion-related gate/up projection variants\n")
    lines.append(f"- Graph 数量：{len(specs)}\n")
    lines.append(f"- 每图候选数：{len(plans)}\n")
    lines.append(f"- Backend：PyTorch eager CUDA\n")
    lines.append(f"- Device：{device_name}\n")
    lines.append(f"- dtype：{args.dtype}\n")
    lines.append(f"- batch_size：1\n")
    lines.append(f"- seq_len：{seq_lens}\n")
    lines.append(f"- hidden_dim：{hidden_dims}\n")
    lines.append(f"- intermediate_dim：{inter_dims}\n")
    lines.append(f"- warmup / measure：{args.warmup} / {args.measure}\n")
    lines.append(f"- selector decision repeats：{args.decision_repeats}\n")
    lines.append("- 偏差说明：第一版未接真实编译器 pass manager；用等价 PyTorch MLP 计算图变体代表 candidate rewrite plans，用于最小体验实验。\n")

    lines.append("\n## 2. 主表 1：整体结果\n")
    lines.append(
        markdown_table(
            selector_rows,
            [
                "Selector",
                "Median Speedup",
                "Geomean Speedup",
                "Median Regret",
                "P90 Regret",
                "Win Rate",
                "P50 Decision Time (ms)",
                "P95 Decision Time (ms)",
            ],
        )
    )

    lines.append("\n## 3. 主表 2：按 shape 分组结果\n")
    lines.append(
        markdown_table(
            shape_rows,
            [
                "Shape",
                "Selector",
                "Median Speedup",
                "Median Regret",
                "Win Rate",
                "P50 Decision Time (ms)",
            ],
        )
    )

    lines.append("\n## 4. Candidate 差异与稳定性\n")
    lines.append(f"- Median candidate spread：{fmt(diversity['median_candidate_spread'])}\n")
    lines.append(f"- P90 candidate spread：{fmt(diversity['p90_candidate_spread'])}\n")
    lines.append(f"- Graphs with >2% candidate spread：{fmt(diversity['graphs_with_over_2pct_spread'])}\n")
    lines.append(f"- Median latency CV：{fmt(diversity['median_latency_cv'])}\n")
    lines.append(f"- P90 latency CV：{fmt(diversity['p90_latency_cv'])}\n")

    lines.append("\n## 5. 图文件\n")
    lines.append("- speedup_distribution.png：Heuristic selector 的 speedup 分布\n")
    lines.append("- regret_distribution.png：Heuristic selector 的 regret 分布\n")

    lines.append("\n## 6. 结论\n")
    if heuristic_row:
        lines.append(
            f"- Candidate plans 是否存在稳定 latency 差异：{'是' if diversity['graphs_with_over_2pct_spread'] >= 0.55 and diversity['p90_latency_cv'] < 0.10 else '不充分'}。"
            f"候选 spread 中位数 {fmt(diversity['median_candidate_spread'])}，P90 CV {fmt(diversity['p90_latency_cv'])}。\n"
        )
        lines.append(
            f"- Heuristic selector 是否优于 baseline：{'是' if heuristic_row['Median Speedup'] > 1.0 and heuristic_row['Win Rate'] > 0.5 else '否/不明显'}。"
            f"Median speedup {fmt(heuristic_row['Median Speedup'])}，win rate {fmt(heuristic_row['Win Rate'])}。\n"
        )
        lines.append(
            f"- Regret 是否可接受：{'是' if heuristic_row['Median Regret'] <= 0.10 else '偏高'}。"
            f"Median regret {fmt(heuristic_row['Median Regret'])}，P90 regret {fmt(heuristic_row['P90 Regret'])}。\n"
        )
        lines.append(
            f"- Decision time 是否足够低：{'是' if heuristic_row['P50 Decision Time (ms)'] <= 10.0 else '否'}。"
            f"P50 {fmt(heuristic_row['P50 Decision Time (ms)'])} ms，P95 {fmt(heuristic_row['P95 Decision Time (ms)'])} ms。\n"
        )
    else:
        lines.append("- Heuristic 结果缺失，无法判断。\n")

    if pass_strong:
        next_step = "达到更强门槛，值得进入下一阶段：加入更真实的 IR/rewrite 枚举，并增加 baseline。"
    elif pass_basic:
        next_step = "达到第一版成功门槛，值得进入下一阶段，但需要用真实 IR/pass manager 复核。"
    else:
        next_step = "未达到第一版成功门槛，下一步应先诊断 candidate diversity、profiling 稳定性或 heuristic 特征，而不是直接上 GNN。"
    lines.append(f"- 是否值得进入下一阶段：{next_step}\n")

    path.write_text("".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("/pub/data/hjwz/rewrite/rewrite_miniexp"))
    parser.add_argument("--graphs-per-shape", type=int, default=20)
    parser.add_argument("--seq-lens", type=str, default="128,512,2048")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measure", type=int, default=30)
    parser.add_argument("--decision-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="Run 1 graph per shape with tiny measurements.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.graphs_per_shape = 1
        args.warmup = min(args.warmup, 2)
        args.measure = min(args.measure, 3)
        args.decision_repeats = min(args.decision_repeats, 10)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiling experiment.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = dtype_from_name(args.dtype)

    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",") if x.strip()]
    specs = make_graph_specs(seq_lens, args.graphs_per_shape, args.dtype, args.seed)
    plans = candidate_plans()

    raw_rows: List[Dict[str, object]] = []
    print(
        f"Running {len(specs)} graphs x {len(plans)} candidates, "
        f"warmup={args.warmup}, measure={args.measure}, device={torch.cuda.get_device_name(torch.cuda.current_device())}",
        flush=True,
    )

    for graph_idx, spec in enumerate(specs, start=1):
        print(
            f"[{graph_idx}/{len(specs)}] {spec.graph_id}: seq={spec.seq_len}, "
            f"h={spec.hidden_dim}, i={spec.intermediate_dim}",
            flush=True,
        )
        tensors = make_tensors(spec, device, dtype)
        # One dry run forces CUDA context setup outside measured rows for this graph.
        with torch.inference_mode():
            _ = run_candidate(plans[0], tensors, spec.intermediate_dim)
            torch.cuda.synchronize()
        for plan in plans:
            rows = measure_candidate(spec, plan, tensors, args.warmup, args.measure)
            raw_rows.extend(rows)
        del tensors
        gc.collect()
        torch.cuda.empty_cache()

    raw_path = args.out_dir / "raw_profile.csv"
    summary_path = args.out_dir / "candidate_summary.csv"
    selection_path = args.out_dir / "selection_result.csv"
    report_path = args.out_dir / "REPORT.md"

    write_csv(raw_path, RAW_FIELDS, raw_rows)
    summaries = summarize_candidates(raw_rows)
    write_csv(summary_path, SUMMARY_FIELDS, summaries)
    selection_rows = compute_selection_results(specs, plans, summaries, args.decision_repeats)
    write_csv(selection_path, SELECTION_FIELDS, selection_rows)

    heuristic_rows = [r for r in selection_rows if r["selector_name"] == "Heuristic"]
    draw_histogram_png(
        args.out_dir / "speedup_distribution.png",
        [float(r["speedup_vs_baseline"]) for r in heuristic_rows],
        bins=20,
        color=(66, 133, 180),
    )
    draw_histogram_png(
        args.out_dir / "regret_distribution.png",
        [float(r["regret_vs_oracle"]) for r in heuristic_rows],
        bins=20,
        color=(221, 132, 82),
    )
    write_report(report_path, args, specs, plans, summaries, selection_rows)

    print(f"Wrote {raw_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {selection_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
