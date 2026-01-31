"""Inspect rhythm/pause features with diagnostics plots."""

from __future__ import annotations

import argparse
import pathlib

import librosa
import numpy as np

from feature_extraction.feature_families import rhythm_pauses


DATA_ROOT = pathlib.Path("datasets/cv-corpus-24.0-2025-12-05/nb-NO/clips")
DEFAULT_FILENAME = "common_voice_nb-NO_42466889.mp3"
OUTPUT_DIR = pathlib.Path("outputs")


def _load_audio(path: pathlib.Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio, sr


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect rhythm/pause features.")
    parser.add_argument("path", nargs="?", default=str(DATA_ROOT / DEFAULT_FILENAME))
    parser.add_argument("--frame-length", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--top-db", type=float, default=40.0)
    args = parser.parse_args()

    audio_path = pathlib.Path(args.path)
    audio, sr = _load_audio(audio_path)

    features = rhythm_pauses.extract(
        audio,
        sr,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        top_db=args.top_db,
    )

    print("--- Rhythm/Pause ---")
    for key in (
        "rhythm_pause_ratio",
        "rhythm_pause_count",
        "rhythm_pause_mean_s",
        "rhythm_onset_rate_hz",
    ):
        print(f"{key}: {features.get(key)}")

    intervals = librosa.effects.split(
        audio,
        top_db=args.top_db,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
    )
    duration = audio.shape[0] / sr if sr > 0 else 0.0

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = audio_path.stem
    output_path = OUTPUT_DIR / f"inspect_rhythm_pauses_{stem}.png"

    times = np.linspace(0, duration, num=audio.size)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
    ax.plot(times, audio, color="0.2", linewidth=0.8)
    ax.set(title="Waveform with nonsilent intervals", xlabel="Time (s)")

    for start, end in intervals:
        ax.axvspan(start / sr, end / sr, color="tab:green", alpha=0.2)
    if intervals.shape[0] >= 2:
        gaps = intervals[1:, 0] - intervals[:-1, 1]
        gap_starts = intervals[:-1, 1]
        for gap, gap_start in zip(gaps, gap_starts):
            if gap > 0:
                ax.axvspan(
                    gap_start / sr,
                    (gap_start + gap) / sr,
                    color="tab:red",
                    alpha=0.15,
                )

    text = (
        f"pause_ratio={features.get('rhythm_pause_ratio'):.2f}\n"
        f"pause_count={features.get('rhythm_pause_count'):.0f}\n"
        f"pause_mean_s={features.get('rhythm_pause_mean_s')}\n"
        f"onset_rate_hz={features.get('rhythm_onset_rate_hz'):.2f}"
    )
    ax.text(0.99, 0.95, text, transform=ax.transAxes, ha="right", va="top")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
