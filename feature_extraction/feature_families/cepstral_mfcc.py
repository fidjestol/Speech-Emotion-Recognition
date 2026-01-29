"""Batch MFCC feature extraction for IEMOCAP.

Reads iemocap_full_dataset.csv, filters valid labels, extracts MFCC-related
features per wav, and writes a single CSV (one row per wav).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd


N_FFT = 1024
HOP_LENGTH = 256
TARGET_SR = 16_000
N_MFCC_LIST = (13, 20, 40)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio, sr


def summarize_matrix(prefix: str, matrix: np.ndarray) -> dict[str, float]:
    stats: dict[str, float] = {}
    stats[f"{prefix}_frames"] = float(matrix.shape[1])
    stats[f"{prefix}_mean"] = float(matrix.mean())
    stats[f"{prefix}_std"] = float(matrix.std())
    stats[f"{prefix}_min"] = float(matrix.min())
    stats[f"{prefix}_max"] = float(matrix.max())
    stats[f"{prefix}_median"] = float(np.median(matrix))
    return stats


def summarize_per_coeff(prefix: str, matrix: np.ndarray) -> dict[str, float]:
    stats: dict[str, float] = {}
    coeff_means = matrix.mean(axis=1)
    coeff_stds = matrix.std(axis=1)
    coeff_mins = matrix.min(axis=1)
    coeff_maxs = matrix.max(axis=1)
    coeff_medians = np.median(matrix, axis=1)
    for idx in range(matrix.shape[0]):
        stats[f"{prefix}_c{idx:02d}_mean"] = float(coeff_means[idx])
        stats[f"{prefix}_c{idx:02d}_std"] = float(coeff_stds[idx])
        stats[f"{prefix}_c{idx:02d}_min"] = float(coeff_mins[idx])
        stats[f"{prefix}_c{idx:02d}_max"] = float(coeff_maxs[idx])
        stats[f"{prefix}_c{idx:02d}_median"] = float(coeff_medians[idx])
    return stats


def extract_mfcc_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    features: dict[str, float] = {}
    duration_s = float(audio.shape[0] / sr)
    features["duration_s"] = duration_s

    for n_mfcc in N_MFCC_LIST:
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            fmin=50,
            fmax=sr // 2,
        )
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        prefix = f"mfcc{n_mfcc}"
        features.update(summarize_matrix(prefix, mfcc))
        features.update(summarize_per_coeff(prefix, mfcc))

        features.update(summarize_matrix(f"{prefix}_d1", delta))
        features.update(summarize_per_coeff(f"{prefix}_d1", delta))

        features.update(summarize_matrix(f"{prefix}_d2", delta2))
        features.update(summarize_per_coeff(f"{prefix}_d2", delta2))

    return features


def iter_rows(df: pd.DataFrame) -> Iterable[dict]:
    for _, row in df.iterrows():
        yield row.to_dict()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Extract MFCC features for IEMOCAP.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=repo_root / "datasets/IEMOCAP/iemocap_full_dataset.csv",
        help="Path to iemocap_full_dataset.csv",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=repo_root / "datasets" / "IEMOCAP",
        help="Root folder containing IEMOCAP audio files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "extracted_features" / "mfcc",
        help="Output directory for MFCC CSV.",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default="mfcc_features.csv",
        help="Output CSV filename.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()

    df = df[(df["emotion"] != "xxx") & (df["agreement"] > 0)].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / args.out_file

    rows: list[dict[str, float | str | int]] = []
    missing: list[str] = []

    for row in iter_rows(df):
        rel_path = row["path"]
        audio_path = args.audio_root / rel_path
        if not audio_path.exists():
            missing.append(str(audio_path))
            continue

        audio, sr = load_audio(audio_path)
        features = extract_mfcc_features(audio, sr)

        record: dict[str, float | str | int] = {
            "path": str(rel_path),
            "session": int(row["session"]),
            "method": row["method"],
            "gender": row["gender"],
            "emotion": row["emotion"],
            "n_annotators": int(row["n_annotators"]),
            "agreement": int(row["agreement"]),
        }
        record.update(features)
        rows.append(record)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)

    print(f"Saved features to: {output_path}")
    if missing:
        print(f"Missing audio files: {len(missing)}")


if __name__ == "__main__":
    main()
