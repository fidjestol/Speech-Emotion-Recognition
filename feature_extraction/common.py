from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

DEFAULT_MACBOOK_MACHINE_NAME = "macbook"
DEFAULT_GENERIC_MACHINE_NAME = "default"


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists():
            return path
    raise FileNotFoundError("Could not locate repository root (missing pyproject.toml).")


def detect_machine_name() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return DEFAULT_MACBOOK_MACHINE_NAME
    return DEFAULT_GENERIC_MACHINE_NAME


def machine_name_from_env(default: str | None = None) -> str:
    raw = str(os.getenv("SER_MACHINE", "")).strip().lower()
    if raw:
        return raw
    if default is not None:
        return str(default).strip().lower() or detect_machine_name()
    return detect_machine_name()


def is_macbook_machine(machine_name: str | None = None) -> bool:
    return machine_name_from_env(default=machine_name) == DEFAULT_MACBOOK_MACHINE_NAME


def logical_cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def _parse_linux_physical_cores() -> int | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return None

    pairs: set[tuple[str, str]] = set()
    physical_id = None
    core_id = None
    with cpuinfo.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                if physical_id is not None and core_id is not None:
                    pairs.add((physical_id, core_id))
                physical_id = None
                core_id = None
                continue
            if line.startswith("physical id"):
                physical_id = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core_id = line.split(":", 1)[1].strip()

    if physical_id is not None and core_id is not None:
        pairs.add((physical_id, core_id))
    if pairs:
        return len(pairs)
    return None


def physical_cpu_count() -> int:
    if platform.system().lower() == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                check=True,
                capture_output=True,
                text=True,
            )
            return max(1, int(result.stdout.strip()))
        except Exception:
            pass

    linux_physical = _parse_linux_physical_cores()
    if linux_physical is not None:
        return max(1, linux_physical)

    return max(1, logical_cpu_count() // 2)


def _reserved_workers(total: int, *, reserved: int = 2, cap: int | None = None) -> int:
    workers = max(1, total - reserved)
    if cap is not None:
        workers = min(workers, cap)
    return max(1, workers)


def resolve_thread_workers(machine_name: str | None = None) -> int:
    if is_macbook_machine(machine_name):
        return _reserved_workers(physical_cpu_count(), reserved=2, cap=8)
    return max(1, logical_cpu_count() - 2)


def resolve_process_workers(machine_name: str | None = None) -> int:
    if is_macbook_machine(machine_name):
        return _reserved_workers(physical_cpu_count(), reserved=2, cap=6)
    return max(1, logical_cpu_count() // 2)


def resolve_transcript_workers(machine_name: str | None = None) -> int:
    if is_macbook_machine(machine_name):
        return _reserved_workers(physical_cpu_count(), reserved=2, cap=4)
    return min(8, logical_cpu_count())


def configure_cpu_math_threads(
    machine_name: str | None = None,
    *,
    num_threads: int | None = None,
) -> int:
    threads = int(num_threads or resolve_thread_workers(machine_name))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(threads))
    return threads


def resolve_torch_device(
    machine_name: str | None = None,
    *,
    requested_device: str | None = None,
) -> str:
    if requested_device is not None:
        return requested_device
    if is_macbook_machine(machine_name):
        return "cpu"

    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_bert_batch_size(
    machine_name: str | None = None,
    *,
    requested_batch_size: int | None = None,
) -> int:
    if requested_batch_size is not None:
        return int(requested_batch_size)
    if is_macbook_machine(machine_name):
        return 16
    return 64


def resolve_ssl_batch_size(
    machine_name: str | None = None,
    *,
    requested_batch_size: int | None = None,
) -> int:
    if requested_batch_size is not None:
        return int(requested_batch_size)
    if is_macbook_machine(machine_name):
        return 4
    return 8


def resolve_ssl_prefetch_workers(
    machine_name: str | None = None,
    *,
    requested_workers: int = 4,
) -> int:
    if is_macbook_machine(machine_name):
        return 1
    return max(1, min(int(requested_workers), logical_cpu_count()))


def should_enable_cuda_amp(
    machine_name: str | None = None,
    *,
    requested: bool = True,
    device: str | None = None,
) -> bool:
    resolved_device = device or resolve_torch_device(machine_name)
    return bool(requested and resolved_device == "cuda")
