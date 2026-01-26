"""Rhythm and pause feature extraction."""

from __future__ import annotations

import numpy as np
import librosa


def _pause_stats(pause_durations: np.ndarray) -> dict[str, float]:
    """Summarize pause durations with NaN-safe stats."""

    if pause_durations.size == 0:
        return {
            "rhythm_pause_count": 0.0,
            "rhythm_pause_mean_s": float("nan"),
            "rhythm_pause_std_s": float("nan"),
            "rhythm_pause_median_s": float("nan"),
            "rhythm_pause_max_s": float("nan"),
        }

    return {
        "rhythm_pause_count": float(pause_durations.size),
        "rhythm_pause_mean_s": float(np.mean(pause_durations)),
        "rhythm_pause_std_s": float(np.std(pause_durations)),
        "rhythm_pause_median_s": float(np.median(pause_durations)),
        "rhythm_pause_max_s": float(np.max(pause_durations)),
    }


def extract(
    audio: np.ndarray,
    sr: int,
    *,
    frame_length: int = 2048,
    hop_length: int = 512,
    top_db: float = 40.0,
) -> dict[str, float]:
    """Extract rhythm and pause statistics from audio."""

    if sr <= 0:
        duration = 0.0
    else:
        duration = float(audio.shape[0]) / float(sr)

    if duration <= 0.0:
        return {
            "rhythm_pause_ratio": 0.0,
            "rhythm_speech_ratio": 0.0,
            "rhythm_pause_count": 0.0,
            "rhythm_pause_mean_s": float("nan"),
            "rhythm_pause_std_s": float("nan"),
            "rhythm_pause_median_s": float("nan"),
            "rhythm_pause_max_s": float("nan"),
            "rhythm_onset_rate_hz": 0.0,
            "rhythm_onset_count": 0.0,
        }

    intervals = librosa.effects.split(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    speech_samples = np.sum(intervals[:, 1] - intervals[:, 0]) if intervals.size else 0
    speech_duration = float(speech_samples) / float(sr)
    total_silence = max(duration - speech_duration, 0.0)
    pause_ratio = total_silence / duration if duration > 0.0 else 0.0

    if intervals.shape[0] >= 2:
        gaps = intervals[1:, 0] - intervals[:-1, 1]
        gaps = gaps[gaps > 0]
        pause_durations = gaps.astype(float) / float(sr)
    else:
        pause_durations = np.array([], dtype=float)

    onsets = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        hop_length=hop_length,
    )
    onset_count = float(onsets.size)
    onset_rate = onset_count / duration if duration > 0.0 else 0.0

    features = {
        "rhythm_pause_ratio": float(pause_ratio),
        "rhythm_speech_ratio": float(1.0 - pause_ratio),
        "rhythm_onset_rate_hz": float(onset_rate),
        "rhythm_onset_count": float(onset_count),
    }
    features.update(_pause_stats(pause_durations))
    return features
