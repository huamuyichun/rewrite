from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
from torch.fx import GraphModule, symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp


LOWERING_FINGERPRINT_VERSION = "inductor-ir-v2"


def _target_name(target: Any) -> str:
    if isinstance(target, str):
        return target
    return getattr(target, "__qualname__", getattr(target, "__name__", str(target)))


def _argument(value: Any, names: dict[Any, str]) -> Any:
    if value in names if isinstance(value, torch.fx.Node) else False:
        return names[value]
    if isinstance(value, (list, tuple)):
        return [_argument(item, names) for item in value]
    if isinstance(value, dict):
        return {str(key): _argument(item, names) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def high_level_fingerprint(module: torch.nn.Module, example: torch.Tensor) -> dict[str, Any]:
    graph_module: GraphModule = symbolic_trace(module)
    ShapeProp(graph_module).propagate(example)
    names = {node: f"n{index}" for index, node in enumerate(graph_module.graph.nodes)}
    nodes: list[dict[str, Any]] = []
    for node in graph_module.graph.nodes:
        tensor_meta = node.meta.get("tensor_meta")
        shape = list(tensor_meta.shape) if hasattr(tensor_meta, "shape") else None
        dtype = str(tensor_meta.dtype).replace("torch.", "") if hasattr(tensor_meta, "dtype") else None
        nodes.append(
            {
                "name": names[node],
                "op": node.op,
                "target": _target_name(node.target),
                "args": _argument(node.args, names),
                "kwargs": _argument(node.kwargs, names),
                "shape": shape,
                "dtype": dtype,
            }
        )
    canonical = json.dumps(nodes, sort_keys=True, separators=(",", ":"))
    return {
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "num_nodes": len(nodes),
        "nodes": nodes,
    }


def _normalize_artifact_text(text: str) -> str:
    text = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
    text = re.sub(r"/pub/data/hjwz/[^\s'\"]+", "/WORKSPACE/PATH", text)
    text = re.sub(r"/tmp/[^\s'\"]+", "/TMP/PATH", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?\b", "TIMESTAMP", text)
    return text


def _hash_parts(parts: list[tuple[str, str]]) -> str | None:
    if not parts:
        return None
    payload = "\n".join(f"## {name}\n{content}" for name, content in sorted(parts))
    return hashlib.sha256(payload.encode()).hexdigest()


def fingerprint_inductor_artifacts(root: Path) -> dict[str, Any]:
    lowered_names = {"ir_pre_fusion.txt", "ir_post_fusion.txt"}
    code_names = {"output_code.py"}
    lowered_parts: list[tuple[str, str]] = []
    code_parts: list[tuple[str, str]] = []
    files: list[str] = []
    execution_lines: list[str] = []

    if root.exists():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = str(path.relative_to(root))
            files.append(relative)
            if path.name not in lowered_names | code_names:
                continue
            text = _normalize_artifact_text(path.read_text(encoding="utf-8", errors="replace"))
            if path.name in lowered_names:
                lowered_parts.append((path.name, text))
            if path.name in code_names:
                code_parts.append((path.name, text))
                for line in text.splitlines():
                    stripped = line.strip()
                    if any(token in stripped for token in ("async_compile.triton", "extern_kernels.", ".run(")):
                        execution_lines.append(stripped)

    execution_text = "\n".join(sorted(execution_lines))
    return {
        "fingerprint_schema_version": LOWERING_FINGERPRINT_VERSION,
        "artifact_files": files,
        "lowered_sha256": _hash_parts(lowered_parts),
        "generated_code_sha256": _hash_parts(code_parts),
        "execution_sha256": hashlib.sha256(execution_text.encode()).hexdigest() if execution_lines else None,
        "execution_records": execution_lines,
    }

