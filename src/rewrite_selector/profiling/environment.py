from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from typing import Any

import torch

try:
    import pynvml
except ImportError:  # pragma: no cover - exercised only in minimal CPU installs
    pynvml = None


def _run(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def physical_gpu_index() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        token = visible.split(",")[0].strip()
        if token.isdigit():
            return int(token)
    return int(torch.cuda.current_device()) if torch.cuda.is_available() else -1


def _decode(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _nvml_snapshot() -> dict[str, Any]:
    if pynvml is None:
        raise RuntimeError("nvidia-ml-py is not installed")
    pynvml.nvmlInit()
    index = physical_gpu_index()
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    processes: list[dict[str, Any]] = []
    try:
        running = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    except pynvml.NVMLError:
        running = []
    for process in running:
        processes.append(
            {
                "pid": int(process.pid),
                "name": "",
                "used_memory_mib": (
                    float(process.usedGpuMemory) / (1024 * 1024)
                    if process.usedGpuMemory is not None
                    else None
                ),
            }
        )
    snapshot = {
        "timestamp_ns": time.time_ns(),
        "cuda_available": True,
        "backend": "nvml",
        "physical_index": index,
        "uuid": _decode(pynvml.nvmlDeviceGetUUID(handle)),
        "name": _decode(pynvml.nvmlDeviceGetName(handle)),
        "temperature_c": int(
            pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        ),
        "power_w": float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0,
        "sm_clock_mhz": int(
            pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        ),
        "memory_clock_mhz": int(
            pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        ),
        "gpu_utilization_percent": int(utilization.gpu),
        "memory_used_mib": float(memory.used) / (1024 * 1024),
        "memory_total_mib": float(memory.total) / (1024 * 1024),
        "compute_processes": processes,
    }
    snapshot["foreign_processes"] = [
        process for process in processes if process["pid"] != os.getpid()
    ]
    return snapshot


def _nvidia_smi_snapshot() -> dict[str, Any]:
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
            processes.append(
                {
                    "pid": int(parts[0]),
                    "name": parts[1],
                    "used_memory_mib": parts[2],
                }
            )
    return {
        "timestamp_ns": time.time_ns(),
        "cuda_available": True,
        "backend": "nvidia_smi",
        "physical_index": index,
        "gpu_csv": output,
        "compute_processes": processes,
        "foreign_processes": [
            process for process in processes if process["pid"] != os.getpid()
        ],
    }


def gpu_snapshot(backend: str = "nvml") -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "timestamp_ns": time.time_ns(),
            "cuda_available": False,
            "backend": "none",
            "compute_processes": [],
            "foreign_processes": [],
        }
    if backend == "nvml":
        try:
            return _nvml_snapshot()
        except Exception as exc:
            fallback = _nvidia_smi_snapshot()
            fallback["nvml_error"] = f"{type(exc).__name__}: {exc}"
            return fallback
    if backend == "nvidia_smi":
        return _nvidia_smi_snapshot()
    raise ValueError(f"unsupported monitor backend: {backend}")


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
        "path_config": {
            key: os.environ.get(key, "")
            for key in (
                "REWRITE_ROOT",
                "REWRITE_ARTIFACT_ROOT",
                "REWRITE_REGISTRY_PATH",
                "QWEN_MODEL_DIR",
                "TMPDIR",
                "XDG_CACHE_HOME",
                "HF_HOME",
            )
        },
        "driver": _run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
        "gpu": gpu_snapshot("nvml"),
    }
    return json.loads(json.dumps(manifest, default=str))


def is_contaminated(snapshots: list[dict[str, Any]]) -> bool:
    return any(snapshot.get("foreign_processes") for snapshot in snapshots)
