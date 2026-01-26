"""Spectrogram representation inspection utility for a Common Voice sample.

Run this module directly to process
`datasets/cv-corpus-24.0-2025-12-05/nb-NO/clips/common_voice_nb-NO_42466889.mp3`
and display both terminal stats and a saved visualization.

Why this exists
----------------
Log-mel spectrograms are a common representation for Speech Emotion Recognition
models. This script keeps the workflow self-contained and adds pooled summaries
for quick analysis.
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass

import librosa
import numpy as np



# -------- Configuration ----------------------------------------------------

DATA_ROOT = pathlib.Path("datasets/cv-corpus-24.0-2025-12-05/nb-NO/clips")
DEFAULT_FILENAME = "common_voice_nb-NO_42466889.mp3"
OUTPUT_DIR = pathlib.Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class RepresentationResult:
    """Container for artefacts we want to report and plot."""

    sr: int
    audio: np.ndarray
    mel: np.ndarray
    log_mel: np.ndarray
    hop_length: int
    n_fft: int
    n_mels: int
    fmin: int
    fmax: int

    @property
    def duration(self) -> float:
        return self.audio.shape[0] / self.sr


# -------- Core pipeline ----------------------------------------------------

def load_audio(path: pathlib.Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    """Load audio and resample to a stable rate."""

    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio, sr


def compute_log_mel(
    audio: np.ndarray,
    sr: int,
    *,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 64,
    fmin: int = 50,
    fmax: int | None = None,
) -> RepresentationResult:
    """Compute a log-mel spectrogram representation."""

    if fmax is None:
        fmax = sr // 2

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    return RepresentationResult(
        sr=sr,
        audio=audio,
        mel=mel,
        log_mel=log_mel,
        hop_length=hop_length,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )


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


def summary_dict(result: RepresentationResult) -> dict[str, float | int | None]:
    """Build a JSON-friendly summary for downstream analysis."""

    log_mel = result.log_mel
    summary: dict[str, float | int | None] = {
        "sample_rate": result.sr,
        "duration_s": result.duration,
        "frames_total": int(log_mel.shape[1]),
        "n_mels": result.n_mels,
        "n_fft": result.n_fft,
        "hop_length": result.hop_length,
        "fmin": result.fmin,
        "fmax": result.fmax,
        "log_mel_mean_db": float(np.mean(log_mel)),
        "log_mel_std_db": float(np.std(log_mel)),
        "log_mel_min_db": float(np.min(log_mel)),
        "log_mel_max_db": float(np.max(log_mel)),
    }
    band_means = log_mel.mean(axis=1)
    band_stds = log_mel.std(axis=1)
    for idx, (mean_val, std_val) in enumerate(zip(band_means, band_stds)):
        summary[f"mel_band{idx:02d}_mean_db"] = float(mean_val)
        summary[f"mel_band{idx:02d}_std_db"] = float(std_val)

    summary = {key: _to_serializable(value) for key, value in summary.items()}
    return summary


def print_summary(result: RepresentationResult) -> None:
    """Emit concise stats to the terminal so runs can be inspected quickly."""

    print("--- Audio ---")
    print(f"Sample rate: {result.sr} Hz")
    print(f"Duration: {result.duration:.2f} s")
    print(f"RMS amplitude: {librosa.feature.rms(y=result.audio).mean():.4f}")

    print("\n--- Log-mel spectrogram ---")
    print(f"Shape (n_mels, frames): {result.log_mel.shape}")
    print(f"Mean log-mel: {result.log_mel.mean():.1f} dB")
    print(f"Std log-mel: {result.log_mel.std():.1f} dB")
    band_means = result.log_mel.mean(axis=1)
    for idx, value in enumerate(band_means[:5]):
        print(f"  Band {idx:02d} mean: {value:.1f} dB")


def plot_features(result: RepresentationResult, output_path: pathlib.Path) -> None:
    """Create a two-panel figure: waveform + log-mel spectrogram."""
    import matplotlib
    # Use a non-interactive backend so plots can be generated in headless runs.
    matplotlib.use("Agg")
    import librosa.display as ldisplay
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

    # Waveform
    axes[0].plot(np.linspace(0, result.duration, num=result.audio.size), result.audio)
    axes[0].set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")

    # Log-mel spectrogram
    img = ldisplay.specshow(
        result.log_mel,
        x_axis="time",
        y_axis="mel",
        sr=result.sr,
        hop_length=result.hop_length,
        ax=axes[1],
        cmap="magma",
    )
    axes[1].set(title="Log-mel spectrogram")
    fig.colorbar(img, ax=axes[1], format="%.1f")

    fig.suptitle("Representation inspection: common_voice_nb-NO_42466889", fontsize=12)
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
    result = compute_log_mel(audio, sr)
    print_summary(result)

    output_path = OUTPUT_DIR / f"representation_{audio_path.stem}.png"
    if plot:
        plot_features(result, output_path)
        print(f"\nSaved visualization to: {output_path}")

    summary = summary_dict(result)
    if ndjson_path is None:
        summary_path = OUTPUT_DIR / f"representation_{audio_path.stem}_summary.json"
        save_summary(summary, summary_path)
        print(f"Saved summary to: {summary_path}")
    else:
        save_summary(summary, None, ndjson_path=ndjson_path)
        print(f"Appended summary to: {ndjson_path}")
    return output_path


if __name__ == "__main__":
    process_file()
