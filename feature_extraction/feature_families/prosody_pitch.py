"""Pitch inspection utility for a Common Voice sample (prosody feature).

Run this module directly to process
`datasets/cv-corpus-24.0-2025-12-05/nb-NO/clips/common_voice_nb-NO_42466889.mp3`
and display both terminal stats and a saved visualization.

Why this exists
----------------
Pitch is a core prosodic cue for Speech Emotion Recognition (SER). A quick,
repeatable script makes it easy to sanity-check voiced/unvoiced detection and
see whether F0 contours look plausible before wiring features into a model.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use a non-interactive backend so plots can be generated in headless runs.
matplotlib.use("Agg")


# -------- Configuration ----------------------------------------------------

DATA_ROOT = pathlib.Path("datasets/cv-corpus-24.0-2025-12-05/nb-NO/clips")
DEFAULT_FILENAME = "common_voice_nb-NO_42466889.mp3"
OUTPUT_DIR = pathlib.Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class PitchResult:
    """Container for artefacts we want to report and plot."""

    sr: int           # target sample rate after loading/resampling
    audio: np.ndarray # mono waveform samples
    f0: np.ndarray    # fundamental frequency in Hz (NaN where unvoiced)
    hop_length: int
    fmin: float
    fmax: float

    @property
    def duration(self) -> float:
        return self.audio.shape[0] / self.sr


# -------- Core pipeline ----------------------------------------------------

def load_audio(path: pathlib.Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    """Load audio and resample to a stable rate.

    Why: Downstream pitch tracking depends on the sampling rate; locking it to
    16 kHz keeps hop lengths and analysis windows consistent across samples.
    """

    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio, sr


def compute_pitch(
    audio: np.ndarray,
    sr: int,
    *,
    fmin: float = 50.0,
    fmax: float = 500.0,
    frame_length: int = 1024,
    hop_length: int = 256,
) -> PitchResult:
    """Compute the pitch contour using YIN and mark unvoiced frames.

    Why: F0 contour shape (range + variability) is a key prosodic cue for SER,
    while voiced/unvoiced regions help avoid spurious statistics.
    """

    f0 = librosa.yin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    return PitchResult(
        sr=sr,
        audio=audio,
        f0=f0,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
    )


# -------- Reporting and visualisation -------------------------------------

def print_summary(result: PitchResult) -> None:
    """Emit concise stats to the terminal so runs can be inspected quickly."""

    print("--- Audio ---")
    print(f"Sample rate: {result.sr} Hz")
    print(f"Duration: {result.duration:.2f} s")
    print(f"RMS amplitude: {librosa.feature.rms(y=result.audio).mean():.4f}")

    f0_valid = result.f0[~np.isnan(result.f0)]
    voiced_ratio = f0_valid.size / max(1, result.f0.size)

    print("\n--- Pitch (F0) ---")
    print(f"Frames (total): {result.f0.size}")
    print(f"Voiced ratio: {voiced_ratio:.2%}")
    if f0_valid.size == 0:
        print("No voiced frames detected.")
        return

    print(f"Mean F0: {f0_valid.mean():.1f} Hz")
    print(f"Median F0: {np.median(f0_valid):.1f} Hz")
    print(f"F0 range: {f0_valid.min():.1f}–{f0_valid.max():.1f} Hz")
    p10, p90 = np.percentile(f0_valid, [10, 90])
    print(f"F0 10–90%: {p10:.1f}–{p90:.1f} Hz")


def plot_features(result: PitchResult, output_path: pathlib.Path) -> None:
    """Create a two-panel figure: waveform + pitch contour."""

    times = librosa.times_like(result.f0, sr=result.sr, hop_length=result.hop_length)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

    # Waveform
    axes[0].plot(np.linspace(0, result.duration, num=result.audio.size), result.audio)
    axes[0].set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")

    # Pitch contour
    axes[1].plot(times, result.f0, color="tab:orange")
    axes[1].set(
        title="Pitch (F0) contour",
        xlabel="Time (s)",
        ylabel="Frequency (Hz)",
        ylim=(max(0.0, result.fmin - 20.0), result.fmax + 20.0),
    )
    axes[1].grid(alpha=0.3)

    fig.suptitle("Pitch inspection: common_voice_nb-NO_42466889", fontsize=12)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# -------- Entrypoint -------------------------------------------------------

def process_file(filename: str = DEFAULT_FILENAME) -> pathlib.Path:
    """End-to-end processing for a single file; returns plot path."""

    audio_path = DATA_ROOT / filename
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    audio, sr = load_audio(audio_path)
    result = compute_pitch(audio, sr)
    print_summary(result)

    output_path = OUTPUT_DIR / f"pitch_{audio_path.stem}.png"
    plot_features(result, output_path)
    print(f"\nSaved visualization to: {output_path}")
    return output_path


if __name__ == "__main__":
    process_file()
