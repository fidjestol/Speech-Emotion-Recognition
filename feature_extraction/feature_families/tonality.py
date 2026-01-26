"""Tonality feature extraction."""

from __future__ import annotations

import librosa
import numpy as np

from utils.stats import summarize


def extract(
    audio: np.ndarray,
    sr: int,
    *,
    hop_length: int = 512,
    n_fft: int = 2048,
    stats: tuple[str, ...] = ("mean", "std", "p10", "p50", "p90"),
) -> dict[str, float]:
    """Extract chroma and tonnetz summaries from audio."""

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft,
    )
    harmonic = librosa.effects.harmonic(audio)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)

    features: dict[str, float] = {}
    for idx, curve in enumerate(chroma):
        features.update(summarize(curve, f"chroma_{idx:02d}", stats=stats))
    for idx, curve in enumerate(tonnetz):
        features.update(summarize(curve, f"tonnetz_{idx:02d}", stats=stats))

    return features
