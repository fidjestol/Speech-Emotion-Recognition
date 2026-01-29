"""Inspect voice-quality features with diagnostics plots."""

from __future__ import annotations

import argparse
import pathlib

import librosa
import numpy as np

from feature_extraction.feature_families import voice_quality


DATA_ROOT = pathlib.Path("datasets/cv-corpus-24.0-2025-12-05/nb-NO/clips")
DEFAULT_FILENAME = "common_voice_nb-NO_42466889.mp3"
OUTPUT_DIR = pathlib.Path("outputs")


def _load_audio(path: pathlib.Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio, sr


def _compute_hnr_contour(audio: np.ndarray, sr: int, fmin: float) -> np.ndarray | None:
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        return None

    sound = parselmouth.Sound(audio, sr)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, fmin, 0.1, 1.0)
    hnr_values = call(harmonicity, "To Matrix").values
    return hnr_values.squeeze()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect voice-quality features.")
    parser.add_argument("path", nargs="?", default=str(DATA_ROOT / DEFAULT_FILENAME))
    parser.add_argument("--fmin", type=float, default=60.0)
    args = parser.parse_args()

    audio_path = pathlib.Path(args.path)
    audio, sr = _load_audio(audio_path)

    features = voice_quality.extract(audio, sr, fmin=args.fmin)
    for key, value in features.items():
        print(f"{key}: {value}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = audio_path.stem
    output_path = OUTPUT_DIR / f"inspect_voice_quality_{stem}.png"

    duration = audio.shape[0] / sr if sr > 0 else 0.0
    times = np.linspace(0, duration, num=audio.size)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

    axes[0].plot(times, audio, color="0.2")
    axes[0].set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")

    hnr_contour = _compute_hnr_contour(audio, sr, args.fmin)
    if hnr_contour is not None and hnr_contour.size:
        axes[1].plot(hnr_contour, color="tab:blue")
        axes[1].set(title="HNR contour (Praat)", xlabel="Frame", ylabel="dB")
    else:
        harmonic, percussive = librosa.effects.hpss(audio)
        frame_length = 2048
        hop_length = 512
        harm_rms = librosa.feature.rms(
            y=harmonic,
            frame_length=frame_length,
            hop_length=hop_length,
        )[0]
        perc_rms = librosa.feature.rms(
            y=percussive,
            frame_length=frame_length,
            hop_length=hop_length,
        )[0]
        axes[1].plot(harm_rms, label="Harmonic RMS", color="tab:green")
        axes[1].plot(perc_rms, label="Residual RMS", color="tab:red")
        axes[1].set(title="HPSS energy (proxy)", xlabel="Frame")
        axes[1].legend(frameon=False)

    axes[0].text(
        0.99,
        0.95,
        "\n".join(f"{k}={v}" for k, v in features.items()),
        transform=axes[0].transAxes,
        ha="right",
        va="top",
    )

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
