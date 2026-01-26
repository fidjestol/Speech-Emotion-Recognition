"""Small helpers for summarizing feature curves."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def summarize(
    curve: np.ndarray,
    prefix: str,
    stats: Sequence[str] = ("mean", "std", "p10", "p50", "p90"),
) -> dict[str, float]:
    """Summarize a 1D curve with named statistics while ignoring NaNs."""

    values = np.asarray(curve, dtype=float).ravel()
    values = values[~np.isnan(values)]

    if values.size == 0:
        return {f"{prefix}_{stat}": float("nan") for stat in stats}

    summary: dict[str, float] = {}
    for stat in stats:
        key = f"{prefix}_{stat}"
        if stat == "mean":
            summary[key] = float(np.mean(values))
        elif stat == "std":
            summary[key] = float(np.std(values))
        elif stat.startswith("p") and stat[1:].isdigit():
            percentile = int(stat[1:])
            summary[key] = float(np.percentile(values, percentile))
        else:
            raise ValueError(f"Unknown stat: {stat}")
    return summary
