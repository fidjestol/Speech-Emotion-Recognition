"""Shared helpers for optional multithreaded feature extraction."""

from __future__ import annotations

import os
import platform
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def available_workers(max_workers: int | None = None) -> int:
    """Return a safe worker count based on CPU availability and optional cap."""

    cpu_count = os.cpu_count() or 1
    if max_workers is None:
        machine_name = str(os.getenv("SER_MACHINE", "")).strip().lower()
        is_macbook = machine_name == "macbook"
        if not machine_name:
            is_macbook = platform.system().lower() == "darwin" and platform.machine().lower() in {
                "arm64",
                "aarch64",
            }
        if is_macbook:
            return max(1, min(cpu_count - 2, 8))
        return max(1, cpu_count)
    return max(1, min(cpu_count, int(max_workers)))


def threaded_map(
    items: Iterable[T],
    worker_fn: Callable[[T], R],
    *,
    max_workers: int | None = None,
    use_threads_if_available: bool = True,
) -> list[R]:
    """Map items with optional threading, preserving input order."""

    inputs = list(items)
    if not inputs:
        return []

    workers = available_workers(max_workers=max_workers)
    if not use_threads_if_available or workers <= 1 or len(inputs) == 1:
        return [worker_fn(item) for item in inputs]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker_fn, inputs))
