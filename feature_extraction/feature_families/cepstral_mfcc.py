"""MFCC feature inspection utility for the MELD train sample.

Run this module directly to process
`datasets/MELD/train/dia0_utt0.flac`
and display both terminal stats and a saved visualization.

Why this exists
----------------
When tuning hyperparameters it is useful to quickly **see** what the extracted MFCCs look
like and verify that the audio decodes as expected. This script keeps the
workflow self‑contained (load → transform → summarise → visualise) with inline
explanations so future contributors can reason about each step.
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
import json
import math

import librosa
import numpy as np



# -------- Configuration ----------------------------------------------------

DATA_ROOT = pathlib.Path("datasets/MELD/train")
DEFAULT_FILENAME: str | None = None
AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".m4a", ".ogg"}
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


def plot_features(
    result: MFCCResult,
    output_path: pathlib.Path,
    title: str,
) -> None:
    """Create a two-panel figure: waveform + MFCC heatmap.

    Why: Visual inspection helps spot clipping, silence, or unusual spectral
    patterns that could mislead the model.
    """
    import matplotlib
    # Use a non-interactive backend so plots can be generated in headless runs.
    matplotlib.use("Agg")
    import librosa.display as ldisplay
    import matplotlib.pyplot as plt

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

    fig.suptitle(title, fontsize=12)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _to_serializable(value: float | int | None) -> float | int | None:
    """Convert values to JSON-safe scalars."""

    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    return value


def summary_dict(result: MFCCResult) -> dict[str, float | int | None]:
    """Build a JSON-friendly summary for downstream analysis."""

    summary: dict[str, float | int | None] = {
        "sample_rate": result.sr,
        "duration_s": result.duration,
        "n_mfcc": int(result.mfcc.shape[0]),
        "frames_total": int(result.mfcc.shape[1]),
    }

    mfcc_means = result.mfcc.mean(axis=1)
    mfcc_stds = result.mfcc.std(axis=1)
    for idx, (mean_val, std_val) in enumerate(zip(mfcc_means, mfcc_stds)):
        summary[f"mfcc_c{idx:02d}_mean"] = float(mean_val)
        summary[f"mfcc_c{idx:02d}_std"] = float(std_val)

    delta_abs_means = np.abs(result.delta).mean(axis=1)
    delta2_abs_means = np.abs(result.delta2).mean(axis=1)
    for idx, (d1_val, d2_val) in enumerate(zip(delta_abs_means, delta2_abs_means)):
        summary[f"mfcc_delta_c{idx:02d}_mean_abs"] = float(d1_val)
        summary[f"mfcc_delta2_c{idx:02d}_mean_abs"] = float(d2_val)

    summary = {key: _to_serializable(value) for key, value in summary.items()}
    return summary


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

def find_first_audio_file(data_root: pathlib.Path) -> pathlib.Path:
    """Return the lexicographically first audio file in a directory."""

    candidates = [
        path
        for path in data_root.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if not candidates:
        raise FileNotFoundError(f"No audio files found in {data_root}")
    return sorted(candidates, key=lambda path: path.name)[0]


def process_file(
    filename: str | None = DEFAULT_FILENAME,
    data_root: pathlib.Path = DATA_ROOT,
) -> pathlib.Path:
    """End-to-end processing for a single file; returns plot path."""

    if filename:
        audio_path = data_root / filename
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
    else:
        audio_path = find_first_audio_file(data_root)

    audio, sr = load_audio(audio_path)
    result = compute_mfcc(audio, sr)
    print(f"Source file: {audio_path}")
    print_summary(result)

    output_path = OUTPUT_DIR / f"mfcc_{audio_path.stem}.png"
    title = f"MFCC inspection: {audio_path.stem}"
    plot_features(result, output_path, title)
    print(f"\nSaved visualization to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect MFCCs for a single audio file.")
    parser.add_argument(
        "--data-root",
        type=pathlib.Path,
        default=DATA_ROOT,
        help="Directory containing audio files.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=DEFAULT_FILENAME,
        help="Audio filename within data-root. If omitted, picks first audio file.",
    )
    args = parser.parse_args()
    process_file(filename=args.filename, data_root=args.data_root)
