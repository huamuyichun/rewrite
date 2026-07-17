from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from typing import Any

import torch


def _run(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def physical_gpu_index() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        return visible.split(",")[0].strip()
    return str(torch.cuda.current_device()) if torch.cuda.is_available() else ""


def gpu_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"timestamp_ns": time.time_ns(), "cuda_available": False}
    index = physical_gpu_index()
    query = (
        "index,uuid,name,temperature.gpu,power.draw,clocks.sm,clocks.mem,"
        "utilization.gpu,memory.used,memory.total"
    )
    output = _run(
        [
            "nvidia-smi",
            f"--id={index}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
    )
    process_output = _run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    processes: list[dict[str, Any]] = []
    for line in process_output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            processes.append({"pid": int(parts[0]), "name": parts[1], "used_memory_mib": parts[2]})
    return {
        "timestamp_ns": time.time_ns(),
        "cuda_available": True,
        "physical_index": index,
        "gpu_csv": output,
        "compute_processes": processes,
        "foreign_processes": [process for process in processes if process["pid"] != os.getpid()],
    }


def environment_manifest() -> dict[str, Any]:
    manifest = {
        "timestamp_ns": time.time_ns(),
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR", ""),
        "driver": _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "gpu": gpu_snapshot(),
    }
    return json.loads(json.dumps(manifest, default=str))


def is_contaminated(snapshots: list[dict[str, Any]]) -> bool:
    return any(snapshot.get("foreign_processes") for snapshot in snapshots)

