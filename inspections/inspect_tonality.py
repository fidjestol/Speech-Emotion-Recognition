"""Inspect tonality features with diagnostics plots."""

from __future__ import annotations

import argparse
import pathlib

import librosa
import numpy as np

from feature_extraction.feature_families import tonality


DATA_ROOT = pathlib.Path("datasets/cv-corpus-24.0-2025-12-05/nb-NO/clips")
DEFAULT_FILENAME = "common_voice_nb-NO_42466889.mp3"
OUTPUT_DIR = pathlib.Path("outputs")


def _load_audio(path: pathlib.Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio, sr


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect tonality features.")
    parser.add_argument("path", nargs="?", default=str(DATA_ROOT / DEFAULT_FILENAME))
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-fft", type=int, default=2048)
    args = parser.parse_args()

    audio_path = pathlib.Path(args.path)
    audio, sr = _load_audio(audio_path)

    features = tonality.extract(
        audio,
        sr,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
    )

    print("--- Tonality ---")
    sample_keys = [k for k in features if k.startswith("chroma_")][:6]
    for key in sorted(sample_keys):
        print(f"{key}: {features[key]:.4f}")
    sample_keys = [k for k in features if k.startswith("tonnetz_")][:6]
    for key in sorted(sample_keys):
        print(f"{key}: {features[key]:.4f}")

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
    )
    harmonic = librosa.effects.harmonic(audio)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)

    import matplotlib

    matplotlib.use("Agg")
    import librosa.display as ldisplay
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = audio_path.stem
    output_path = OUTPUT_DIR / f"inspect_tonality_{stem}.png"

    duration = audio.shape[0] / sr if sr > 0 else 0.0
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

    axes[0].plot(np.linspace(0, duration, num=audio.size), audio, color="0.2")
    axes[0].set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")

    img = ldisplay.specshow(
        chroma,
        x_axis="time",
        y_axis="chroma",
        sr=sr,
        hop_length=args.hop_length,
        ax=axes[1],
        cmap="magma",
    )
    axes[1].set(title="Chroma (STFT)")
    fig.colorbar(img, ax=axes[1], format="%.2f")

    for idx in range(tonnetz.shape[0]):
        axes[2].plot(tonnetz[idx], label=f"T{idx}")
    axes[2].set(title="Tonnetz (6 dimensions)", xlabel="Frame")
    axes[2].legend(loc="upper right", frameon=False, ncol=3, fontsize=8)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
