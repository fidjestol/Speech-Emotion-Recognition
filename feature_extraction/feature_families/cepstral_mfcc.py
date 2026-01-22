"""MFCC feature inspection utility for the Common Voice sample.

Run this module directly to process
`datasets/cv-corpus-24.0-2025-12-05/nb-NO/clips/common_voice_nb-NO_42466889.mp3`
and display both terminal stats and a saved visualization.

Why this exists
----------------
When tuning hyperparameters it is useful to quickly **see** what the extracted MFCCs look
like and verify that the audio decodes as expected. This script keeps the
workflow self‑contained (load → transform → summarise → visualise) with inline
explanations so future contributors can reason about each step.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import librosa
import librosa.display as ldisplay
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
class MFCCResult:
    """Container for artefacts we want to report and plot."""

    sr: int              # target sample rate after loading/resampling
    audio: np.ndarray    # mono waveform samples
    mfcc: np.ndarray     # MFCC coefficient matrix (n_mfcc x frames)
    delta: np.ndarray    # first-order temporal derivatives of MFCCs
    delta2: np.ndarray   # second-order temporal derivatives of MFCCs

    @property
    def duration(self) -> float:
        return self.audio.shape[0] / self.sr


# -------- Core pipeline ----------------------------------------------------

def load_audio(path: pathlib.Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    """Load audio and resample to a stable rate.

    Why: Downstream MFCC calculations depend on the sampling rate; locking it to
    16 kHz keeps hop lengths and FFT windows consistent across samples.
    """

    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio, sr


def compute_mfcc(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> MFCCResult:
    """Compute MFCCs plus first/second-order deltas.

    Why: MFCCs capture the spectral envelope; deltas encode local dynamics, both
    of which are common cues for emotion recognition models.
    """

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=50,
        fmax=sr // 2,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return MFCCResult(sr=sr, audio=audio, mfcc=mfcc, delta=delta, delta2=delta2)


# -------- Reporting and visualisation -------------------------------------

def print_summary(result: MFCCResult) -> None:
    """Emit concise stats to the terminal so runs can be inspected quickly."""

    print("--- Audio ---")
    print(f"Sample rate: {result.sr} Hz")
    print(f"Duration: {result.duration:.2f} s")
    print(f"RMS amplitude: {librosa.feature.rms(y=result.audio).mean():.4f}")

    print("\n--- MFCC ---")
    print(f"Shape (n_coeffs, frames): {result.mfcc.shape}")
    print("Coefficient means (first 5):")
    mfcc_means = result.mfcc.mean(axis=1)
    for idx, coeff in enumerate(mfcc_means[:5]):
        print(f"  C{idx:02d}: {coeff:+.3f}")

    print("\nDelta stats (mean absolute, first 5):")
    delta_abs_means = np.abs(result.delta).mean(axis=1)
    for idx, coeff in enumerate(delta_abs_means[:5]):
        print(f"  ΔC{idx:02d}: {coeff:.3f}")


def plot_features(result: MFCCResult, output_path: pathlib.Path) -> None:
    """Create a two-panel figure: waveform + MFCC heatmap.

    Why: Visual inspection helps spot clipping, silence, or unusual spectral
    patterns that could mislead the model.
    """

    times = librosa.times_like(result.mfcc, sr=result.sr, hop_length=256)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

    # Waveform
    axes[0].plot(np.linspace(0, result.duration, num=result.audio.size), result.audio)
    axes[0].set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")

    # MFCC heatmap
    img = ldisplay.specshow(
        result.mfcc,
        x_axis="time",
        sr=result.sr,
        hop_length=256,
        ax=axes[1],
        cmap="magma",
    )
    axes[1].set(title="MFCCs", ylabel="Coefficient index")
    fig.colorbar(img, ax=axes[1], format="%.1f")

    fig.suptitle("MFCC inspection: common_voice_nb-NO_42466889", fontsize=12)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# -------- Entrypoint -------------------------------------------------------

def process_file(filename: str = DEFAULT_FILENAME) -> pathlib.Path:
    """End-to-end processing for a single file; returns plot path."""

    audio_path = DATA_ROOT / filename
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    audio, sr = load_audio(audio_path)
    result = compute_mfcc(audio, sr)
    print_summary(result)

    output_path = OUTPUT_DIR / f"mfcc_{audio_path.stem}.png"
    plot_features(result, output_path)
    print(f"\nSaved visualization to: {output_path}")
    return output_path


if __name__ == "__main__":
    process_file()
