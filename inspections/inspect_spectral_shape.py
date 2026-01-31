"""Inspect spectral-shape features with diagnostics plots."""

from __future__ import annotations

import argparse
import pathlib

import librosa
import numpy as np

from feature_extraction.feature_families import spectral_shape


DATA_ROOT = pathlib.Path("datasets/cv-corpus-24.0-2025-12-05/nb-NO/clips")
DEFAULT_FILENAME = "common_voice_nb-NO_42466889.mp3"
OUTPUT_DIR = pathlib.Path("outputs")


def _load_audio(path: pathlib.Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio, sr


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect spectral-shape features.")
    parser.add_argument("path", nargs="?", default=str(DATA_ROOT / DEFAULT_FILENAME))
    parser.add_argument("--frame-length", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--roll-percent", type=float, default=0.85)
    args = parser.parse_args()

    audio_path = pathlib.Path(args.path)
    audio, sr = _load_audio(audio_path)

    features = spectral_shape.extract(
        audio,
        sr,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
    )
    print("--- Spectral shape ---")
    for key in sorted(features):
        if key.endswith("_mean"):
            print(f"{key}: {features[key]:.3f}")

    stft = librosa.stft(audio, n_fft=args.frame_length, hop_length=args.hop_length)
    magnitude = np.abs(stft)
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(
        S=magnitude,
        sr=sr,
        roll_percent=args.roll_percent,
    )[0]
    flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
    )[0]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = audio_path.stem
    output_path = OUTPUT_DIR / f"inspect_spectral_shape_{stem}.png"

    duration = audio.shape[0] / sr if sr > 0 else 0.0
    times = np.linspace(0, duration, num=audio.size)
    feat_times = librosa.times_like(centroid, sr=sr, hop_length=args.hop_length)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(times, audio, color="0.2")
    axes[0].set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")

    axes[1].plot(feat_times, centroid, label="Centroid", color="tab:blue")
    axes[1].plot(feat_times, rolloff, label="Rolloff", color="tab:orange")
    axes[1].set(title="Centroid / Rolloff", xlabel="Time (s)", ylabel="Hz")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.3)

    axes[2].plot(feat_times, flatness, label="Flatness", color="tab:green")
    axes[2].plot(feat_times, zcr, label="ZCR", color="tab:red")
    axes[2].set(title="Flatness / ZCR", xlabel="Time (s)")
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.3)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
