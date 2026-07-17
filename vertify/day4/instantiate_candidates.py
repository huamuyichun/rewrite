#!/usr/bin/env python3
"""Day 4: instantiate candidate rewrite plans as executable modules and FX graphs.

This script intentionally stops before latency profiling. It verifies that the
Day 3 candidate space can be materialized into executable PyTorch modules,
traced into Torch FX graphs, deduplicated by actual graph structure, and checked
for numerical equivalence against the baseline plan.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import GraphModule, symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp


@dataclass(frozen=True)
class BlockConfig:
    batch_size: int
    seq_len: int
    hidden_dim: int
    intermediate_dim: int
    dtype: str
    seed: int


class SeparateMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        activation: str,
        multiply: str,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.activation = activation
        self.multiply = multiply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.activation == "f_silu":
            hidden_gate = F.silu(gate)
        elif self.activation == "manual_silu":
            hidden_gate = gate * torch.sigmoid(gate)
        else:
            raise RuntimeError(f"unsupported activation: {self.activation}")

        if self.multiply == "out_of_place":
            hidden = hidden_gate * up
        elif self.multiply == "inplace":
            hidden = hidden_gate.mul_(up)
        else:
            raise RuntimeError(f"unsupported multiply: {self.multiply}")
        return self.down_proj(hidden)


class FusedGateUpMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        split_mode: str,
        activation: str,
        multiply: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_dim, 2 * intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.intermediate_dim = intermediate_dim
        self.split_mode = split_mode
        self.activation = activation
        self.multiply = multiply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        if self.split_mode == "chunk":
            gate, up = torch.chunk(gate_up, 2, dim=-1)
        elif self.split_mode == "split":
            gate, up = torch.split(gate_up, self.intermediate_dim, dim=-1)
        else:
            raise RuntimeError(f"unsupported split_mode: {self.split_mode}")

        if self.activation == "f_silu":
            hidden_gate = F.silu(gate)
        elif self.activation == "manual_silu":
            hidden_gate = gate * torch.sigmoid(gate)
        else:
            raise RuntimeError(f"unsupported activation: {self.activation}")

        if self.multiply == "out_of_place":
            hidden = hidden_gate * up
        elif self.multiply == "inplace":
            hidden = hidden_gate.mul_(up)
        else:
            raise RuntimeError(f"unsupported multiply: {self.multiply}")
        return self.down_proj(hidden)


def dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_name(target: Any) -> str:
    if isinstance(target, str):
        return target
    return getattr(target, "__name__", str(target))


def node_shape(node: Any) -> str:
    tensor_meta = node.meta.get("tensor_meta")
    if tensor_meta is None:
        return ""
    if isinstance(tensor_meta, tuple):
        shapes = []
        for item in tensor_meta:
            if hasattr(item, "shape"):
                shapes.append(tuple(item.shape))
            else:
                shapes.append(str(type(item).__name__))
        return str(shapes)
    if hasattr(tensor_meta, "shape"):
        return str(tuple(tensor_meta.shape))
    return str(type(tensor_meta).__name__)


def node_dtype(node: Any) -> str:
    tensor_meta = node.meta.get("tensor_meta")
    if tensor_meta is None:
        return ""
    if isinstance(tensor_meta, tuple):
        dtypes = []
        for item in tensor_meta:
            if hasattr(item, "dtype"):
                dtypes.append(str(item.dtype).replace("torch.", ""))
            else:
                dtypes.append(str(type(item).__name__))
        return str(dtypes)
    if hasattr(tensor_meta, "dtype"):
        return str(tensor_meta.dtype).replace("torch.", "")
    return str(type(tensor_meta).__name__)


def node_rows(graph_module: GraphModule) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, node in enumerate(graph_module.graph.nodes):
        rows.append(
            {
                "node_idx": idx,
                "name": node.name,
                "op": node.op,
                "target": target_name(node.target),
                "args": str(node.args),
                "users": ",".join(user.name for user in node.users),
                "shape": node_shape(node),
                "dtype": node_dtype(node),
            }
        )
    return rows


def graph_signature(graph_module: GraphModule) -> tuple[str, str]:
    items: list[str] = []
    for node in graph_module.graph.nodes:
        arg_names = []
        for arg in node.args:
            if hasattr(arg, "name"):
                arg_names.append(arg.name)
            elif isinstance(arg, (tuple, list)):
                arg_names.append(
                    "(" + ",".join(a.name if hasattr(a, "name") else str(a) for a in arg) + ")"
                )
            else:
                arg_names.append(str(arg))
        users = ",".join(sorted(user.name for user in node.users))
        items.append(f"{node.op}:{target_name(node.target)}:{'|'.join(arg_names)}:{users}")
    text = "\n".join(items)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_baseline(cfg: BlockConfig, device: torch.device, dtype: torch.dtype) -> SeparateMLP:
    model = SeparateMLP(
        hidden_dim=cfg.hidden_dim,
        intermediate_dim=cfg.intermediate_dim,
        activation="f_silu",
        multiply="out_of_place",
    )
    return model.to(device=device, dtype=dtype).eval()


def instantiate_candidate(
    plan: dict[str, Any],
    cfg: BlockConfig,
    baseline: SeparateMLP,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    projection = plan["gate_up_projection"]
    activation = plan["activation"]
    multiply = plan["multiply"]

    if projection == "separate":
        model = SeparateMLP(
            hidden_dim=cfg.hidden_dim,
            intermediate_dim=cfg.intermediate_dim,
            activation=activation,
            multiply=multiply,
        )
        model.to(device=device, dtype=dtype).eval()
        with torch.no_grad():
            model.gate_proj.weight.copy_(baseline.gate_proj.weight)
            model.up_proj.weight.copy_(baseline.up_proj.weight)
            model.down_proj.weight.copy_(baseline.down_proj.weight)
        return model

    if projection == "fused":
        model = FusedGateUpMLP(
            hidden_dim=cfg.hidden_dim,
            intermediate_dim=cfg.intermediate_dim,
            split_mode=plan["gate_up_split"],
            activation=activation,
            multiply=multiply,
        )
        model.to(device=device, dtype=dtype).eval()
        with torch.no_grad():
            model.gate_up_proj.weight.copy_(
                torch.cat([baseline.gate_proj.weight, baseline.up_proj.weight], dim=0)
            )
            model.down_proj.weight.copy_(baseline.down_proj.weight)
        return model

    raise ValueError(f"unsupported gate_up_projection: {projection}")


def equivalence_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    diff = (reference - actual).detach().float().abs()
    ref_abs = reference.detach().float().abs()
    max_abs = float(diff.max().cpu())
    mean_abs = float(diff.mean().cpu())
    denom = torch.clamp(ref_abs, min=1e-6)
    max_rel = float((diff / denom).max().cpu())
    return {
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "max_rel_diff": max_rel,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, default=Path(__file__).resolve().parents[1] / "day3" / "candidate_plans.json")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--intermediate-dim", type=int, default=4096)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--rtol", type=float, default=2e-2)
    args = parser.parse_args()

    cfg = BlockConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        dtype=args.dtype,
        seed=args.seed,
    )

    set_seed(cfg.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = out_dir / "candidate_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(cfg.dtype)
    plans = json.loads(args.plans.read_text(encoding="utf-8"))

    baseline = make_baseline(cfg, device, dtype)
    example = torch.randn(
        cfg.batch_size,
        cfg.seq_len,
        cfg.hidden_dim,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        baseline_output = baseline(example)

    summary_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    signature_to_candidates: dict[str, list[str]] = {}

    for plan in plans:
        candidate_id = plan["candidate_id"]
        candidate = instantiate_candidate(plan, cfg, baseline, device, dtype)
        traced = symbolic_trace(candidate)
        ShapeProp(traced).propagate(example)
        rows = node_rows(traced)
        signature_hash, signature_text = graph_signature(traced)
        signature_to_candidates.setdefault(signature_hash, []).append(candidate_id)

        candidate_dir = graph_dir / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "fx_graph_code.py").write_text(traced.code, encoding="utf-8")
        (candidate_dir / "fx_graph_readable.txt").write_text(str(traced.graph), encoding="utf-8")
        (candidate_dir / "actual_graph_signature.txt").write_text(signature_text, encoding="utf-8")
        write_csv(candidate_dir / "fx_nodes.csv", rows)
        (candidate_dir / "fx_nodes.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

        with torch.no_grad():
            candidate_output = candidate(example)
        metrics = equivalence_metrics(baseline_output, candidate_output)
        allclose = bool(
            torch.allclose(
                baseline_output.float(),
                candidate_output.float(),
                atol=args.atol,
                rtol=args.rtol,
            )
        )

        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "is_baseline": plan["is_baseline"],
                "rewrite_family": plan["rewrite_family"],
                "plan_desc": plan["plan_desc"],
                "day3_structural_signature": plan["structural_signature"],
                "actual_signature_hash": signature_hash,
                "num_fx_nodes": len(rows),
                "fx_graph_dir": str(candidate_dir),
            }
        )
        equivalence_rows.append(
            {
                "candidate_id": candidate_id,
                "allclose_to_baseline": allclose,
                "atol": args.atol,
                "rtol": args.rtol,
                **metrics,
            }
        )

    duplicated = {k: v for k, v in signature_to_candidates.items() if len(v) > 1}
    dedup_result = {
        "status": "ok" if not duplicated else "has_duplicates",
        "num_candidates": len(plans),
        "num_unique_actual_signatures": len(signature_to_candidates),
        "duplicated_actual_signatures": duplicated,
    }
    equivalence_result = {
        "status": "ok" if all(row["allclose_to_baseline"] for row in equivalence_rows) else "failed",
        "num_candidates": len(equivalence_rows),
        "num_passed": sum(1 for row in equivalence_rows if row["allclose_to_baseline"]),
        "atol": args.atol,
        "rtol": args.rtol,
    }
    metadata = {
        "purpose": "Day 4 candidate instantiation and equivalence check",
        "config": asdict(cfg),
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "candidate_plan_source": str(args.plans),
        "dedup_status": dedup_result["status"],
        "equivalence_status": equivalence_result["status"],
    }

    write_csv(out_dir / "candidate_summary.csv", summary_rows)
    write_csv(out_dir / "equivalence_results.csv", equivalence_rows)
    (out_dir / "candidate_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    (out_dir / "equivalence_results.json").write_text(json.dumps(equivalence_rows, indent=2), encoding="utf-8")
    (out_dir / "dedup_result.json").write_text(json.dumps(dedup_result, indent=2), encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    result = {
        "metadata": metadata,
        "dedup_result": dedup_result,
        "equivalence_result": equivalence_result,
    }
    print(json.dumps(result, indent=2))

    if dedup_result["status"] != "ok":
        raise SystemExit(1)
    if equivalence_result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
