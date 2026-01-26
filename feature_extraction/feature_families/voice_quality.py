"""Voice quality feature extraction."""

from __future__ import annotations

import numpy as np


def _hnr_from_parselmouth(
    audio: np.ndarray,
    sr: int,
    *,
    fmin: float,
) -> float | None:
    """Compute HNR via Praat/parselmouth, returning None if unavailable."""

    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        return None

    sound = parselmouth.Sound(audio, sr)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, fmin, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    if isinstance(hnr, float) and np.isfinite(hnr):
        return float(hnr)
    return float("nan")


def _hnr_proxy_from_hpss(audio: np.ndarray) -> float:
    """Compute a proxy HNR using harmonic/percussive energy ratio."""

    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "Voice-quality fallback requires librosa. Install with: uv add librosa"
        ) from exc

    harmonic, percussive = librosa.effects.hpss(audio)
    harm_energy = float(np.mean(harmonic ** 2))
    noise_energy = float(np.mean(percussive ** 2))
    total_energy = harm_energy + noise_energy
    if total_energy <= 0.0:
        return float("nan")
    eps = 1e-12
    return float(10.0 * np.log10((harm_energy + eps) / (noise_energy + eps)))


def extract(
    audio: np.ndarray,
    sr: int,
    *,
    fmin: float = 60.0,
    fmax: float = 400.0,
) -> dict[str, float]:
    """Extract voice-quality features from audio."""

    _ = fmax  # reserved for future jitter/shimmer extraction
    audio = np.asarray(audio, dtype=np.float32)

    hnr = _hnr_from_parselmouth(audio, sr, fmin=fmin)
    if hnr is not None:
        return {"vq_hnr_db": float(hnr)}

    hnr_proxy = _hnr_proxy_from_hpss(audio)
    return {"vq_hnr_proxy_db": float(hnr_proxy)}
