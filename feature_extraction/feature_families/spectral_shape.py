"""Spectral shape inspection utility for a Common Voice sample (SER feature).

Run this module directly to process
`datasets/cv-corpus-24.0-2025-12-05/nb-NO/clips/common_voice_nb-NO_42466889.mp3`
and display both terminal stats and a saved visualization.

Why this exists
----------------
Spectral-shape cues (centroid, rolloff, contrast, etc.) capture timbral changes
that correlate with emotion. This script keeps the workflow self-contained and
adds pooled feature summaries for downstream analysis.
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass

import librosa
import numpy as np

from utils.stats import summarize



# -------- Configuration ----------------------------------------------------

DATA_ROOT = pathlib.Path("datasets/cv-corpus-24.0-2025-12-05/nb-NO/clips")
DEFAULT_FILENAME = "common_voice_nb-NO_42466889.mp3"
OUTPUT_DIR = pathlib.Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class SpectralShapeResult:
    """Container for artefacts we want to report and plot."""

    sr: int
    audio: np.ndarray
    centroid: np.ndarray
    bandwidth: np.ndarray
    rolloff: np.ndarray
    flatness: np.ndarray
    flux: np.ndarray
    contrast: np.ndarray
    zcr: np.ndarray
    hop_length: int
    frame_length: int
    roll_percent: float

    @property
    def duration(self) -> float:
        return self.audio.shape[0] / self.sr


# -------- Core pipeline ----------------------------------------------------

def load_audio(path: pathlib.Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    """Load audio and resample to a stable rate."""

    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio, sr


def _compute_flux(magnitude: np.ndarray) -> np.ndarray:
    """Compute spectral flux from a magnitude spectrogram."""

    if magnitude.shape[1] <= 1:
        return np.full(magnitude.shape[1], np.nan)

    diff = np.diff(magnitude, axis=1)
    flux = np.sqrt(np.sum(diff ** 2, axis=0))
    return np.pad(flux, (1, 0), mode="constant", constant_values=np.nan)


def compute_spectral_shape(
    audio: np.ndarray,
    sr: int,
    *,
    frame_length: int = 2048,
    hop_length: int = 512,
    roll_percent: float = 0.85,
) -> SpectralShapeResult:
    """Compute a set of spectral-shape contours from audio."""

    stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    magnitude = np.abs(stft)

    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(
        S=magnitude,
        sr=sr,
        roll_percent=roll_percent,
    )[0]
    flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
    flux = _compute_flux(magnitude)
    contrast = librosa.feature.spectral_contrast(S=magnitude, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    return SpectralShapeResult(
        sr=sr,
        audio=audio,
        centroid=centroid,
        bandwidth=bandwidth,
        rolloff=rolloff,
        flatness=flatness,
        flux=flux,
        contrast=contrast,
        zcr=zcr,
        hop_length=hop_length,
        frame_length=frame_length,
        roll_percent=roll_percent,
    )


def extract(
    audio: np.ndarray,
    sr: int,
    *,
    frame_length: int = 2048,
    hop_length: int = 512,
    stats: tuple[str, ...] = ("mean", "std", "p10", "p50", "p90"),
) -> dict[str, float]:
    """Return pooled spectral-shape features as a flat dict."""

    result = compute_spectral_shape(
        audio,
        sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    features: dict[str, float] = {}
    features.update(summarize(result.centroid, "spec_centroid", stats))
    features.update(summarize(result.bandwidth, "spec_bandwidth", stats))
    features.update(summarize(result.rolloff, "spec_rolloff", stats))
    features.update(summarize(result.flatness, "spec_flatness", stats))
    features.update(summarize(result.flux, "spec_flux", stats))
    features.update(summarize(result.zcr, "zcr", stats))

    for idx, band in enumerate(result.contrast):
        features.update(summarize(band, f"spec_contrast_b{idx}", stats))

    return features


# -------- Reporting and visualisation -------------------------------------

def _to_serializable(value: float | int | None) -> float | int | None:
    """Convert values to JSON-safe scalars."""

    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    return value


def summary_dict(result: SpectralShapeResult) -> dict[str, float | int | None]:
    """Build a JSON-friendly summary for downstream analysis."""

    summary: dict[str, float | int | None] = {
        "sample_rate": result.sr,
        "duration_s": result.duration,
        "frames_total": int(result.centroid.size),
        "roll_percent": result.roll_percent,
    }
    summary.update(summarize(result.centroid, "spec_centroid"))
    summary.update(summarize(result.bandwidth, "spec_bandwidth"))
    summary.update(summarize(result.rolloff, "spec_rolloff"))
    summary.update(summarize(result.flatness, "spec_flatness"))
    summary.update(summarize(result.flux, "spec_flux"))
    summary.update(summarize(result.zcr, "zcr"))
    for idx, band in enumerate(result.contrast):
        summary.update(summarize(band, f"spec_contrast_b{idx}"))
    summary = {key: _to_serializable(value) for key, value in summary.items()}
    return summary


def print_summary(result: SpectralShapeResult) -> None:
    """Emit concise stats to the terminal so runs can be inspected quickly."""

    print("--- Audio ---")
    print(f"Sample rate: {result.sr} Hz")
    print(f"Duration: {result.duration:.2f} s")
    print(f"RMS amplitude: {librosa.feature.rms(y=result.audio).mean():.4f}")

    print("\n--- Spectral shape ---")
    print(f"Frames (total): {result.centroid.size}")
    print(f"Rolloff percent: {result.roll_percent:.2f}")
    print(f"Centroid mean: {np.nanmean(result.centroid):.1f} Hz")
    print(f"Bandwidth mean: {np.nanmean(result.bandwidth):.1f} Hz")
    print(f"Rolloff mean: {np.nanmean(result.rolloff):.1f} Hz")
    print(f"Flatness mean: {np.nanmean(result.flatness):.3f}")
    print(f"Flux mean: {np.nanmean(result.flux):.3f}")
    print(f"ZCR mean: {np.nanmean(result.zcr):.3f}")


def plot_features(result: SpectralShapeResult, output_path: pathlib.Path) -> None:
    """Create a multi-panel figure with key spectral-shape contours."""
    import matplotlib
    # Use a non-interactive backend so plots can be generated in headless runs.
    matplotlib.use("Agg")
    import librosa.display as ldisplay
    import matplotlib.pyplot as plt

    times = librosa.times_like(result.centroid, sr=result.sr, hop_length=result.hop_length)

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), constrained_layout=True)

    # Waveform
    axes[0].plot(np.linspace(0, result.duration, num=result.audio.size), result.audio)
    axes[0].set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")

    # Spectral centroid + bandwidth
    axes[1].plot(times, result.centroid, label="Centroid", color="tab:blue")
    axes[1].plot(times, result.bandwidth, label="Bandwidth", color="tab:green")
    axes[1].set(title="Centroid & bandwidth", xlabel="Time (s)", ylabel="Hz")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right", frameon=False)

    # Rolloff + flatness
    axes[2].plot(times, result.rolloff, label="Rolloff", color="tab:orange")
    axes[2].plot(times, result.flatness, label="Flatness", color="tab:red")
    axes[2].set(title="Rolloff & flatness", xlabel="Time (s)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper right", frameon=False)

    # Spectral contrast (heatmap)
    img = ldisplay.specshow(
        result.contrast,
        x_axis="time",
        sr=result.sr,
        hop_length=result.hop_length,
        ax=axes[3],
        cmap="magma",
    )
    axes[3].set(title="Spectral contrast", ylabel="Band")
    fig.colorbar(img, ax=axes[3], format="%.1f")

    fig.suptitle("Spectral shape inspection: common_voice_nb-NO_42466889", fontsize=12)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_summary(
    summary: dict[str, float | int | None],
    output_path: pathlib.Path | None,
    *,
    ndjson_path: pathlib.Path | None = None,
) -> None:
    """Write summary metrics to JSON or append to NDJSON."""

    if ndjson_path is not None:
        with ndjson_path.open("a", encoding="utf-8") as handle:
            json.dump(summary, handle, separators=(",", ":"))
            handle.write("\n")
        return

    if output_path is None:
        raise ValueError("output_path is required when ndjson_path is None")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, separators=(",", ":"))


# -------- Entrypoint -------------------------------------------------------

def process_file(
    filename: str = DEFAULT_FILENAME,
    *,
    plot: bool = True,
    ndjson_path: pathlib.Path | None = None,
) -> pathlib.Path:
    """End-to-end processing for a single file; returns plot path."""

    audio_path = DATA_ROOT / filename
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    audio, sr = load_audio(audio_path)
    result = compute_spectral_shape(audio, sr)
    print_summary(result)

    output_path = OUTPUT_DIR / f"spectral_shape_{audio_path.stem}.png"
    if plot:
        plot_features(result, output_path)
        print(f"\nSaved visualization to: {output_path}")

    summary = summary_dict(result)
    if ndjson_path is None:
        summary_path = OUTPUT_DIR / f"spectral_shape_{audio_path.stem}_summary.json"
        save_summary(summary, summary_path)
        print(f"Saved summary to: {summary_path}")
    else:
        save_summary(summary, None, ndjson_path=ndjson_path)
        print(f"Appended summary to: {ndjson_path}")
    return output_path


if __name__ == "__main__":
    process_file()
