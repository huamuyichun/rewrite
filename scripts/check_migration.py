#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from rewrite_selector.profiling.environment import gpu_snapshot


EXPECTED_TORCH = "2.10.0"
PHASE1_SESSIONS = (
    "phase1_qwen_decode_monitor_off_20260717/qwen_s06",
    "phase1_qwen_decode_monitor_off_20260717/qwen_s07",
    "phase1_qwen_decode_monitor_off_20260717/qwen_s08",
)


def _path_from_env(name: str, default: Path | None = None) -> Path | None:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser().resolve()
    return default.resolve() if default is not None else None


def _existing_writable_parent(path: Path) -> Path | None:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current if current.exists() and os.access(current, os.W_OK) else None


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _check_qwen(model_dir: Path | None) -> tuple[bool, dict[str, Any]]:
    if model_dir is None:
        return False, {"reason": "QWEN_MODEL_DIR 未设置"}
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return False, {"path": str(model_dir), "reason": "缺少 config.json"}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    weight_files = sorted(
        path.name
        for pattern in ("*.safetensors", "*.bin")
        for path in model_dir.glob(pattern)
    )
    valid_shape = (
        int(config.get("hidden_size", -1)) == 3584
        and int(config.get("intermediate_size", -1)) == 18944
    )
    return bool(weight_files and valid_shape), {
        "path": str(model_dir),
        "model_type": config.get("model_type"),
        "hidden_size": config.get("hidden_size"),
        "intermediate_size": config.get("intermediate_size"),
        "weight_file_count": len(weight_files),
        "valid_qwen2p5_7b_shape": valid_shape,
    }


def _check_phase1_artifacts(artifact_root: Path) -> tuple[bool, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_ok = True
    for relative in PHASE1_SESSIONS:
        session_dir = artifact_root / "phase1" / relative
        required = [
            session_dir / "environment.json",
            session_dir / "resolved_config.json",
            session_dir / "session_summary.json",
            session_dir / "status.json",
            session_dir
            / "groups"
            / "qwen2p5_7b_decode_bs1_t1_bf16"
            / "result.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        rows.append(
            {
                "session": relative.rsplit("/", 1)[-1],
                "path": str(session_dir),
                "missing": missing,
            }
        )
        all_ok = all_ok and not missing
    return all_ok, {"sessions": rows}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="检查 rewrite 仓库迁移后的 CPU、路径、模型、artifact 与 GPU 门禁"
    )
    parser.add_argument("--require-qwen", action="store_true")
    parser.add_argument("--require-phase1-artifacts", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    artifact_root = _path_from_env(
        "REWRITE_ARTIFACT_ROOT",
        ROOT / "artifacts",
    )
    assert artifact_root is not None
    registry_path = _path_from_env(
        "REWRITE_REGISTRY_PATH",
        ROOT / "artifacts" / "registry.jsonl",
    )
    qwen_dir = _path_from_env("QWEN_MODEL_DIR")
    configured_root = _path_from_env("REWRITE_ROOT", ROOT)

    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, required: bool, details: Any) -> None:
        checks.append(
            {
                "name": name,
                "status": "ok" if ok else ("error" if required else "warning"),
                "details": details,
            }
        )

    add(
        "repository_root",
        configured_root == ROOT.resolve(),
        True,
        {"configured": str(configured_root), "detected": str(ROOT.resolve())},
    )
    add(
        "python_version",
        sys.version_info[:2] == (3, 12),
        True,
        {"version": sys.version},
    )
    torch_base = torch.__version__.split("+", 1)[0]
    add(
        "torch_version",
        torch_base == EXPECTED_TORCH,
        True,
        {
            "version": torch.__version__,
            "expected_base": EXPECTED_TORCH,
            "cuda_runtime": torch.version.cuda,
        },
    )
    add(
        "artifact_root_writable",
        _existing_writable_parent(artifact_root) is not None,
        True,
        {"path": str(artifact_root)},
    )
    assert registry_path is not None
    add(
        "registry_parent_writable",
        _existing_writable_parent(registry_path.parent) is not None,
        True,
        {"path": str(registry_path)},
    )

    git_status = _git("status", "--short")
    add(
        "git_clean",
        not git_status,
        args.require_clean_git,
        {
            "commit": _git("rev-parse", "HEAD"),
            "status": git_status.splitlines(),
        },
    )

    qwen_ok, qwen_details = _check_qwen(qwen_dir)
    add("qwen_weights", qwen_ok, args.require_qwen, qwen_details)

    phase1_ok, phase1_details = _check_phase1_artifacts(artifact_root)
    add(
        "phase1_artifacts",
        phase1_ok,
        args.require_phase1_artifacts,
        phase1_details,
    )

    visible = [
        token.strip()
        for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if token.strip()
    ]
    gpu_details: dict[str, Any] = {
        "cuda_visible_devices": visible,
        "cuda_available": torch.cuda.is_available(),
    }
    gpu_ok = (
        len(visible) == 1
        and visible[0].isdigit()
        and torch.cuda.is_available()
    )
    if gpu_ok:
        gpu_details["device_name"] = torch.cuda.get_device_name(0)
        snapshot = gpu_snapshot("nvml")
        gpu_details["snapshot"] = snapshot
        utilization = snapshot.get("gpu_utilization_percent")
        gpu_ok = (
            not snapshot.get("foreign_processes")
            and (utilization is None or int(utilization) <= 5)
        )
    add("gpu_gate", gpu_ok, args.require_gpu, gpu_details)

    errors = [item for item in checks if item["status"] == "error"]
    report = {
        "schema_version": "rewrite-migration-check-v1",
        "status": "ok" if not errors else "failed",
        "repository_root": str(ROOT.resolve()),
        "artifact_root": str(artifact_root),
        "registry_path": str(registry_path),
        "checks": checks,
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
