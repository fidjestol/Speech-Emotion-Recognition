"""Energy (RMS) inspection utility for a Common Voice sample (prosody feature).

Run this module directly to process
`datasets/cv-corpus-24.0-2025-12-05/nb-NO/clips/common_voice_nb-NO_42466889.mp3`
and display both terminal stats and a saved visualization.

Why this exists
----------------
Energy is a core prosodic cue for Speech Emotion Recognition (SER). This script
keeps the workflow self-contained and makes it easy to sanity-check loudness
contours without leaking energy into silence.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
import json
import math

import librosa
import numpy as np



# -------- Configuration ----------------------------------------------------

DATA_ROOT = pathlib.Path("datasets/cv-corpus-24.0-2025-12-05/nb-NO/clips")
DEFAULT_FILENAME = "common_voice_nb-NO_42466889.mp3"
OUTPUT_DIR = pathlib.Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class EnergyResult:
    """Container for artefacts we want to report and plot."""

    sr: int           # target sample rate after loading/resampling
    audio: np.ndarray # mono waveform samples
    rms: np.ndarray   # RMS energy per frame
    rms_db: np.ndarray
    hop_length: int
    frame_length: int
    rms_db_threshold: float
    rms_db_percentile: float | None
    rms_db_effective_threshold: float
    silence_top_db: float
    min_voiced_frames: int

    @property
    def duration(self) -> float:
        return self.audio.shape[0] / self.sr


# -------- Core pipeline ----------------------------------------------------

def load_audio(path: pathlib.Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    """Load audio and resample to a stable rate."""

    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio, sr


def _drop_short_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    """Drop short True runs shorter than min_run frames."""

    if min_run <= 1:
        return mask

    cleaned = mask.copy()
    run_start = None
    for idx, is_true in enumerate(mask):
        if is_true and run_start is None:
            run_start = idx
        if not is_true and run_start is not None:
            run_len = idx - run_start
            if run_len < min_run:
                cleaned[run_start:idx] = False
            run_start = None
    if run_start is not None:
        run_len = len(mask) - run_start
        if run_len < min_run:
            cleaned[run_start:] = False
    return cleaned


def compute_energy(
    audio: np.ndarray,
    sr: int,
    *,
    frame_length: int = 2048,
    hop_length: int = 256,
    rms_db_threshold: float = -50.0,
    rms_db_percentile: float | None = 10.0,
    silence_top_db: float = 40.0,
    min_voiced_frames: int = 3,
) -> EnergyResult:
    """Compute RMS energy and mask silent frames.

    Strategy:
    - Use RMS as a loudness proxy.
    - Combine an absolute dB threshold with a percentile-based floor so the
      silence gate adapts to the recording.
    - Enforce a nonsilent mask to avoid energy during silence.
    """

    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    if rms_db_percentile is None:
        rms_db_effective = rms_db_threshold
    else:
        rms_db_effective = max(
            rms_db_threshold,
            float(np.percentile(rms_db, rms_db_percentile)),
        )

    energy_mask = rms_db >= rms_db_effective
    nonsilent_intervals = librosa.effects.split(
        audio,
        top_db=silence_top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    nonsilent_mask = np.zeros_like(energy_mask, dtype=bool)
    for start, end in nonsilent_intervals:
        start_frame = start // hop_length
        end_frame = int(np.ceil(end / hop_length))
        nonsilent_mask[start_frame:end_frame] = True
    energy_mask &= nonsilent_mask
    energy_mask = _drop_short_runs(energy_mask, min_voiced_frames)

    rms_db_masked = rms_db.copy()
    rms_db_masked[~energy_mask] = np.nan

    return EnergyResult(
        sr=sr,
        audio=audio,
        rms=rms,
        rms_db=rms_db_masked,
        hop_length=hop_length,
        frame_length=frame_length,
        rms_db_threshold=rms_db_threshold,
        rms_db_percentile=rms_db_percentile,
        rms_db_effective_threshold=rms_db_effective,
        silence_top_db=silence_top_db,
        min_voiced_frames=min_voiced_frames,
    )


# -------- Reporting and visualisation -------------------------------------

def print_summary(result: EnergyResult) -> None:
    """Emit concise stats to the terminal so runs can be inspected quickly."""

    print("--- Audio ---")
    print(f"Sample rate: {result.sr} Hz")
    print(f"Duration: {result.duration:.2f} s")

    rms_valid = result.rms_db[~np.isnan(result.rms_db)]
    voiced_ratio = rms_valid.size / max(1, result.rms_db.size)

    print("\n--- Energy (RMS dB) ---")
    print(f"Frames (total): {result.rms_db.size}")
    print(f"Voiced ratio: {voiced_ratio:.2%}")
    print(f"RMS threshold: {result.rms_db_threshold:.1f} dB")
    if result.rms_db_percentile is not None:
        print(f"RMS percentile: {result.rms_db_percentile:.1f}%")
    print(f"RMS effective: {result.rms_db_effective_threshold:.1f} dB")
    print(f"Silence top_db: {result.silence_top_db:.1f} dB")
    print(f"Min voiced frames: {result.min_voiced_frames}")
    if rms_valid.size == 0:
        print("No voiced frames detected.")
        print("Hint: try lowering thresholds or widening the silence mask.")
        return

    print(f"Mean RMS: {rms_valid.mean():.1f} dB")
    print(f"Median RMS: {np.median(rms_valid):.1f} dB")
    print(f"RMS range: {rms_valid.min():.1f}–{rms_valid.max():.1f} dB")
    p10, p90 = np.percentile(rms_valid, [10, 90])
    print(f"RMS 10–90%: {p10:.1f}–{p90:.1f} dB")


def plot_features(result: EnergyResult, output_path: pathlib.Path) -> None:
    """Create a two-panel figure: waveform + RMS energy contour."""
    import matplotlib

    # Use a non-interactive backend so plots can be generated in headless runs.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = librosa.times_like(result.rms_db, sr=result.sr, hop_length=result.hop_length)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

    # Waveform
    axes[0].plot(np.linspace(0, result.duration, num=result.audio.size), result.audio)
    axes[0].set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")

    # RMS energy (dB)
    axes[1].plot(times, result.rms_db, color="tab:green")
    axes[1].set(
        title="RMS energy (dB)",
        xlabel="Time (s)",
        ylabel="RMS (dB)",
    )
    axes[1].grid(alpha=0.3)

    fig.suptitle("Energy inspection: common_voice_nb-NO_42466889", fontsize=12)
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


def summary_dict(result: EnergyResult) -> dict[str, float | int | None]:
    """Build a JSON-friendly summary for downstream analysis."""

    rms_valid = result.rms_db[~np.isnan(result.rms_db)]
    voiced_ratio = rms_valid.size / max(1, result.rms_db.size)

    summary: dict[str, float | int | None] = {
        "sample_rate": result.sr,
        "duration_s": result.duration,
        "frames_total": int(result.rms_db.size),
        "voiced_ratio": float(voiced_ratio),
        "rms_threshold_db": result.rms_db_threshold,
        "rms_percentile": result.rms_db_percentile,
        "rms_effective_db": result.rms_db_effective_threshold,
        "silence_top_db": result.silence_top_db,
        "min_voiced_frames": result.min_voiced_frames,
    }
    if rms_valid.size:
        p10, p90 = np.percentile(rms_valid, [10, 90])
        summary.update(
            {
                "rms_mean_db": float(rms_valid.mean()),
                "rms_median_db": float(np.median(rms_valid)),
                "rms_min_db": float(rms_valid.min()),
                "rms_max_db": float(rms_valid.max()),
                "rms_p10_db": float(p10),
                "rms_p90_db": float(p90),
            }
        )
    else:
        summary.update(
            {
                "rms_mean_db": None,
                "rms_median_db": None,
                "rms_min_db": None,
                "rms_max_db": None,
                "rms_p10_db": None,
                "rms_p90_db": None,
            }
        )

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
    result = compute_energy(audio, sr)
    print_summary(result)

    output_path = OUTPUT_DIR / f"energy_{audio_path.stem}.png"
    if plot:
        plot_features(result, output_path)
        print(f"\nSaved visualization to: {output_path}")

    summary = summary_dict(result)
    if ndjson_path is None:
        summary_path = OUTPUT_DIR / f"energy_{audio_path.stem}_summary.json"
        save_summary(summary, summary_path)
        print(f"Saved summary to: {summary_path}")
    else:
        save_summary(summary, None, ndjson_path=ndjson_path)
        print(f"Appended summary to: {ndjson_path}")
    return output_path


if __name__ == "__main__":
    process_file()
