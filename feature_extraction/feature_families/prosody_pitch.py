"""Pitch inspection utility for a Common Voice sample (prosody feature).

Run this module directly to process
`datasets/cv-corpus-24.0-2025-12-05/nb-NO/clips/common_voice_nb-NO_42466889.mp3`
and display both terminal stats and a saved visualization.

Why this exists
----------------
Pitch is a core prosodic cue for Speech Emotion Recognition (SER). This script
keeps the workflow self-contained and makes it easy to sanity-check voiced
segments without leaking F0 into silence.
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
class PitchResult:
    """Container for artefacts we want to report and plot."""

    sr: int           # target sample rate after loading/resampling
    audio: np.ndarray # mono waveform samples
    raw_f0: np.ndarray
    voiced_prob: np.ndarray | None
    f0: np.ndarray    # fundamental frequency in Hz (NaN where unvoiced)
    hop_length: int
    fmin: float
    fmax: float
    rms_db_threshold: float
    rms_db_percentile: float | None
    rms_db_effective_threshold: float
    voicing_prob_threshold: float
    min_voicing_prob_mean: float
    silence_top_db: float
    max_gap_frames: int
    min_voiced_frames: int

    @property
    def duration(self) -> float:
        return self.audio.shape[0] / self.sr


# -------- Core pipeline ----------------------------------------------------

def load_audio(path: pathlib.Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    """Load audio and resample to a stable rate."""

    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio, sr


def _fill_short_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Fill gaps of False values up to max_gap_frames in a boolean mask."""

    if max_gap_frames <= 0:
        return mask

    filled = mask.copy()
    gap_start = None
    for idx, is_true in enumerate(mask):
        if not is_true and gap_start is None:
            gap_start = idx
        if is_true and gap_start is not None:
            gap_len = idx - gap_start
            if gap_len <= max_gap_frames:
                filled[gap_start:idx] = True
            gap_start = None
    return filled


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


def compute_pitch(
    audio: np.ndarray,
    sr: int,
    *,
    fmin: float = 50.0,
    fmax: float = 400.0,
    frame_length: int = 4096,
    hop_length: int = 512,
    rms_db_threshold: float = -55.0,
    rms_db_percentile: float | None = 15.0,
    voicing_prob_threshold: float = 0.4,
    silence_top_db: float = 50.0,
    max_gap_frames: int = 4,
    min_voiced_frames: int = 3,
    min_voicing_prob_mean: float = 0.1,
) -> PitchResult:
    """Compute the pitch contour and mask unvoiced/low-energy frames.

    Strategy:
    - Track F0 with pYIN for a stable voicing probability estimate.
    - Mark a frame as voiced if it is energetic OR has high voicing probability.
    - Enforce a nonsilent mask to avoid F0 in silence.
    - Fill short gaps inside speech, but never bridge long silent regions.
    """

    f0, _, voiced_prob = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    if np.nanmean(voiced_prob) < min_voicing_prob_mean:
        # Fall back to YIN if pYIN fails to find stable voicing.
        f0 = librosa.yin(
            audio,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        voiced_prob = None
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

    if voiced_prob is None:
        voiced_mask = rms_db >= rms_db_effective
    else:
        voiced_mask = (rms_db >= rms_db_effective) | (
            voiced_prob >= voicing_prob_threshold
        )

    nonsilent_intervals = librosa.effects.split(
        audio,
        top_db=silence_top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    nonsilent_mask = np.zeros_like(voiced_mask, dtype=bool)
    for start, end in nonsilent_intervals:
        start_frame = start // hop_length
        end_frame = int(np.ceil(end / hop_length))
        nonsilent_mask[start_frame:end_frame] = True
    voiced_mask &= nonsilent_mask

    voiced_mask = _fill_short_gaps(voiced_mask, max_gap_frames)
    voiced_mask = _drop_short_runs(voiced_mask, min_voiced_frames)

    f0_masked = f0.astype(float, copy=True)
    f0_masked[~voiced_mask] = np.nan

    return PitchResult(
        sr=sr,
        audio=audio,
        raw_f0=f0,
        voiced_prob=voiced_prob,
        f0=f0_masked,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        rms_db_threshold=rms_db_threshold,
        rms_db_percentile=rms_db_percentile,
        rms_db_effective_threshold=rms_db_effective,
        voicing_prob_threshold=voicing_prob_threshold,
        min_voicing_prob_mean=min_voicing_prob_mean,
        silence_top_db=silence_top_db,
        max_gap_frames=max_gap_frames,
        min_voiced_frames=min_voiced_frames,
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
    raw_f0_valid = result.raw_f0[~np.isnan(result.raw_f0)]
    raw_voiced_ratio = raw_f0_valid.size / max(1, result.raw_f0.size)

    print("\n--- Pitch (F0) ---")
    print(f"Frames (total): {result.f0.size}")
    print(f"Voiced ratio: {voiced_ratio:.2%}")
    print(f"Raw voiced ratio: {raw_voiced_ratio:.2%}")
    print(f"RMS threshold: {result.rms_db_threshold:.1f} dB")
    if result.rms_db_percentile is not None:
        print(f"RMS percentile: {result.rms_db_percentile:.1f}%")
    print(f"RMS effective: {result.rms_db_effective_threshold:.1f} dB")
    print(f"Min voicing prob mean: {result.min_voicing_prob_mean:.2f}")
    if result.voiced_prob is not None:
        print(f"Voicing prob threshold: {result.voicing_prob_threshold:.2f}")
    print(f"Silence top_db: {result.silence_top_db:.1f} dB")
    print(f"Max gap frames: {result.max_gap_frames}")
    print(f"Min voiced frames: {result.min_voiced_frames}")
    if raw_f0_valid.size:
        print(f"Raw F0 range: {raw_f0_valid.min():.1f}–{raw_f0_valid.max():.1f} Hz")
    if result.voiced_prob is not None:
        print(
            "Voicing prob (min/mean/max): "
            f"{np.nanmin(result.voiced_prob):.2f}/"
            f"{np.nanmean(result.voiced_prob):.2f}/"
            f"{np.nanmax(result.voiced_prob):.2f}"
        )
    if f0_valid.size == 0:
        print("No voiced frames detected.")
        print("Hint: try lowering thresholds or widening the F0 range.")
        return

    print(f"Mean F0: {f0_valid.mean():.1f} Hz")
    print(f"Median F0: {np.median(f0_valid):.1f} Hz")
    print(f"F0 range: {f0_valid.min():.1f}–{f0_valid.max():.1f} Hz")
    p10, p90 = np.percentile(f0_valid, [10, 90])
    print(f"F0 10–90%: {p10:.1f}–{p90:.1f} Hz")


def plot_features(
    result: PitchResult,
    output_path: pathlib.Path,
    *,
    interpolate_for_plot: bool = False,
    max_gap_frames: int = 3,
) -> None:
    """Create a three-panel figure: waveform, pitch contour, voicing probability.

    If interpolation is enabled, it is only for visualization, and only fills
    short gaps inside speech. Long silent regions remain as gaps.
    """
    import matplotlib

    # Use a non-interactive backend so plots can be generated in headless runs.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = librosa.times_like(result.f0, sr=result.sr, hop_length=result.hop_length)
    f0_plot = result.f0.copy()
    if interpolate_for_plot:
        # Visualization-only interpolation; analysis always uses NaN-masked F0.
        isnan = np.isnan(f0_plot)
        if np.any(~isnan):
            start = None
            for idx, missing in enumerate(isnan):
                if missing and start is None:
                    start = idx
                if not missing and start is not None:
                    gap = idx - start
                    if gap <= max_gap_frames and start > 0:
                        f0_plot[start:idx] = np.linspace(
                            f0_plot[start - 1],
                            f0_plot[idx],
                            num=gap + 2,
                        )[1:-1]
                    start = None

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

    # Waveform
    axes[0].plot(np.linspace(0, result.duration, num=result.audio.size), result.audio)
    axes[0].set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")

    # Pitch contour
    axes[1].plot(times, result.raw_f0, color="0.7", linewidth=1.0, label="Raw F0")
    axes[1].plot(times, f0_plot, color="tab:orange", label="Masked F0")
    axes[1].set(
        title="Pitch (F0) contour",
        xlabel="Time (s)",
        ylabel="Frequency (Hz)",
        ylim=(max(0.0, result.fmin - 20.0), result.fmax + 20.0),
    )
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right", frameon=False)

    # Voicing probability
    if result.voiced_prob is not None:
        axes[2].plot(times, result.voiced_prob, color="tab:purple")
        axes[2].axhline(
            result.voicing_prob_threshold,
            color="0.4",
            linestyle="--",
            linewidth=1.0,
        )
        axes[2].set(title="Voicing probability", xlabel="Time (s)", ylabel="Prob")
        axes[2].set_ylim(0.0, 1.0)
        axes[2].grid(alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "Voicing probability not computed", ha="center")
        axes[2].set_axis_off()

    fig.suptitle("Pitch inspection: common_voice_nb-NO_42466889", fontsize=12)
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


def summary_dict(result: PitchResult) -> dict[str, float | int | None]:
    """Build a JSON-friendly summary for downstream analysis."""

    f0_valid = result.f0[~np.isnan(result.f0)]
    voiced_ratio = f0_valid.size / max(1, result.f0.size)
    raw_f0_valid = result.raw_f0[~np.isnan(result.raw_f0)]
    raw_voiced_ratio = raw_f0_valid.size / max(1, result.raw_f0.size)

    summary: dict[str, float | int | None] = {
        "sample_rate": result.sr,
        "duration_s": result.duration,
        "frames_total": int(result.f0.size),
        "voiced_ratio": float(voiced_ratio),
        "raw_voiced_ratio": float(raw_voiced_ratio),
        "rms_threshold_db": result.rms_db_threshold,
        "rms_percentile": result.rms_db_percentile,
        "rms_effective_db": result.rms_db_effective_threshold,
        "min_voicing_prob_mean": result.min_voicing_prob_mean,
        "voicing_prob_threshold": result.voicing_prob_threshold
        if result.voiced_prob is not None
        else None,
        "silence_top_db": result.silence_top_db,
        "max_gap_frames": result.max_gap_frames,
        "min_voiced_frames": result.min_voiced_frames,
    }

    if raw_f0_valid.size:
        summary.update(
            {
                "raw_f0_min_hz": float(raw_f0_valid.min()),
                "raw_f0_max_hz": float(raw_f0_valid.max()),
            }
        )
    else:
        summary.update({"raw_f0_min_hz": None, "raw_f0_max_hz": None})

    if result.voiced_prob is not None:
        summary.update(
            {
                "voicing_prob_min": float(np.nanmin(result.voiced_prob)),
                "voicing_prob_mean": float(np.nanmean(result.voiced_prob)),
                "voicing_prob_max": float(np.nanmax(result.voiced_prob)),
            }
        )
    else:
        summary.update(
            {
                "voicing_prob_min": None,
                "voicing_prob_mean": None,
                "voicing_prob_max": None,
            }
        )

    if f0_valid.size:
        p10, p90 = np.percentile(f0_valid, [10, 90])
        summary.update(
            {
                "f0_mean_hz": float(f0_valid.mean()),
                "f0_median_hz": float(np.median(f0_valid)),
                "f0_min_hz": float(f0_valid.min()),
                "f0_max_hz": float(f0_valid.max()),
                "f0_p10_hz": float(p10),
                "f0_p90_hz": float(p90),
            }
        )
    else:
        summary.update(
            {
                "f0_mean_hz": None,
                "f0_median_hz": None,
                "f0_min_hz": None,
                "f0_max_hz": None,
                "f0_p10_hz": None,
                "f0_p90_hz": None,
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
    result = compute_pitch(audio, sr)
    print_summary(result)

    output_path = OUTPUT_DIR / f"pitch_{audio_path.stem}.png"
    if plot:
        plot_features(result, output_path)
        print(f"\nSaved visualization to: {output_path}")

    summary = summary_dict(result)
    if ndjson_path is None:
        summary_path = OUTPUT_DIR / f"pitch_{audio_path.stem}_summary.json"
        save_summary(summary, summary_path)
        print(f"Saved summary to: {summary_path}")
    else:
        save_summary(summary, None, ndjson_path=ndjson_path)
        print(f"Appended summary to: {ndjson_path}")
    return output_path


if __name__ == "__main__":
    process_file()
