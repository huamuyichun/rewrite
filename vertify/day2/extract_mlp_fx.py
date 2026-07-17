#!/usr/bin/env python3
"""Day 2: extract a minimal Transformer MLP block as a Torch FX graph.

This script does not enumerate rewrite candidates and does not profile latency.
It only verifies that the chosen Day 1 block can be represented as a stable
Torch FX graph with enough structure for later candidate generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp


@dataclass(frozen=True)
class BlockConfig:
    batch_size: int
    seq_len: int
    hidden_dim: int
    intermediate_dim: int
    dtype: str
    seed: int


class SwiGLUMLP(nn.Module):
    """Minimal LLM-style MLP block: gate/up projections, activation, down projection."""

    def __init__(self, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = F.silu(gate) * up
        return self.down_proj(hidden)


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized == "fp16":
        return torch.float16
    if normalized == "bf16":
        return torch.bfloat16
    if normalized == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def node_shape(node: Any) -> str:
    tensor_meta = node.meta.get("tensor_meta")
    if tensor_meta is None:
        return ""
    return str(tuple(tensor_meta.shape))


def node_dtype(node: Any) -> str:
    tensor_meta = node.meta.get("tensor_meta")
    if tensor_meta is None:
        return ""
    return str(tensor_meta.dtype).replace("torch.", "")


def target_name(target: Any) -> str:
    if isinstance(target, str):
        return target
    return getattr(target, "__name__", str(target))


def extract_node_rows(graph_module: torch.fx.GraphModule) -> list[dict[str, Any]]:
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--intermediate-dim", type=int, default=4096)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--seed", type=int, default=20260525)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(cfg.dtype)

    model = SwiGLUMLP(cfg.hidden_dim, cfg.intermediate_dim).to(device=device, dtype=dtype).eval()
    example = torch.randn(
        cfg.batch_size,
        cfg.seq_len,
        cfg.hidden_dim,
        device=device,
        dtype=dtype,
    )

    traced = symbolic_trace(model)
    ShapeProp(traced).propagate(example)
    rows = extract_node_rows(traced)

    with torch.no_grad():
        eager_out = model(example)
        fx_out = traced(example)
        max_abs_diff = float((eager_out - fx_out).abs().max().detach().cpu())

    (out_dir / "fx_graph_code.py").write_text(traced.code, encoding="utf-8")
    (out_dir / "fx_graph_readable.txt").write_text(str(traced.graph), encoding="utf-8")
    write_csv(out_dir / "fx_nodes.csv", rows)
    (out_dir / "fx_nodes.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    metadata = {
        "purpose": "Day 2 Torch FX extraction for minimal Transformer MLP block",
        "config": asdict(cfg),
        "python": str(Path(torch.__file__).resolve()),
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "num_fx_nodes": len(rows),
        "max_abs_diff_eager_vs_fx": max_abs_diff,
        "status": "ok" if max_abs_diff == 0.0 else "check_diff",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
