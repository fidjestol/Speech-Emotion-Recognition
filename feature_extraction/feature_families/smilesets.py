"""openSMILE standard feature-set extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_SMILE_CACHE: dict[str, Any] = {}


def _require_opensmile() -> Any:
    """Import opensmile with a clear error message if missing."""

    try:
        import opensmile
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "openSMILE features require the opensmile package. "
            "Install with: uv add opensmile"
        ) from exc
    return opensmile


def _get_smile(feature_set: str) -> Any:
    """Load or reuse a Smile extractor for the requested feature set."""

    cached = _SMILE_CACHE.get(feature_set)
    if cached is not None:
        return cached

    opensmile = _require_opensmile()
    feature_map = {
        "GeMAPSv01b": opensmile.FeatureSet.GeMAPSv01b,
        "eGeMAPSv02": opensmile.FeatureSet.eGeMAPSv02,
        "ComParE_2016": opensmile.FeatureSet.ComParE_2016,
    }
    if feature_set not in feature_map:
        valid = ", ".join(sorted(feature_map))
        raise ValueError(f"Unknown feature_set '{feature_set}'. Valid: {valid}")

    smile = opensmile.Smile(
        feature_set=feature_map[feature_set],
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    _SMILE_CACHE[feature_set] = smile
    return smile


def extract(
    audio_path: str | Path,
    *,
    feature_set: str = "eGeMAPSv02",
) -> dict[str, float]:
    """Extract openSMILE functional features from an audio file."""

    smile = _get_smile(feature_set)
    features = smile.process_file(str(audio_path))

    flat: dict[str, float] = {}
    for name, value in features.iloc[0].items():
        flat[f"smile_{name}"] = float(value)
    return flat
