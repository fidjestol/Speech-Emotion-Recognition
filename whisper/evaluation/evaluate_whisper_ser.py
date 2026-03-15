from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_ID = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
TARGET_LABELS = ["ang", "fea", "hap", "neu", "sad", "sur"]
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.20
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_DURATION = 30.0

IEMOCAP_TO_EVAL_LABEL = {
    "ang": "ang",
    "fea": "fea",
    "hap": "hap",
    "neu": "neu",
    "sad": "sad",
    "sur": "sur",
}

# This mirrors the Whisper checkpoint's native label names back into the repo's label space.
MODEL_TO_IEMOCAP_LABEL = {
    "angry": "ang",
    "disgust": "dis",
    "fearful": "fea",
    "happy": "hap",
    "neutral": "neu",
    "sad": "sad",
    "surprised": "sur",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Whisper SER checkpoint on IEMOCAP using the same "
            "80/20 stratified test split and metric style used in feature_selection notebooks."
        )
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on test rows for smoke runs.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Force the inference device. Default uses CUDA when available.",
    )
    parser.add_argument(
        "--output-dir",
        default="whisper/evaluation/artifacts/whisper_large_v3_iemocap_clean_6way",
        help="Where evaluation outputs should be written.",
    )
    return parser.parse_args()


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not find repository root from current location.")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower()


def build_eval_frame(repo_root: Path) -> pd.DataFrame:
    metadata_path = repo_root / "datasets" / "IEMOCAP" / "iemocap_full_dataset.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"IEMOCAP metadata not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    df["emotion"] = normalize_series(df["emotion"])
    df["method"] = normalize_series(df["method"])
    df["gender"] = df["gender"].astype(str).str.strip().str.upper()
    df["path"] = df["path"].astype(str).str.replace("\\", "/", regex=False).str.strip()

    label_map = dict(IEMOCAP_TO_EVAL_LABEL)
    df = df[df["emotion"].isin(label_map)].copy()
    df = df[df["agreement"] > 0].copy()
    df["eval_label"] = df["emotion"].map(label_map)
    df["audio_path"] = df["path"].map(lambda rel: str((repo_root / "datasets" / "IEMOCAP" / rel).resolve()))
    df["audio_exists"] = df["audio_path"].map(lambda p: Path(p).exists())
    df = df[df["audio_exists"]].reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No evaluation rows remained after filtering and audio-path checks.")

    return df


def load_audio(audio_path: Path, target_sr: int, max_duration: float) -> np.ndarray:
    audio, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    max_length = int(target_sr * max_duration)
    if len(audio) >= max_length:
        return audio[:max_length].astype(np.float32, copy=False)

    padded = np.zeros(max_length, dtype=np.float32)
    padded[: len(audio)] = audio.astype(np.float32, copy=False)
    return padded


def batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def run_inference(
    model: AutoModelForAudioClassification,
    feature_extractor: AutoFeatureExtractor,
    frame: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    max_duration: float,
) -> tuple[list[str], np.ndarray]:
    audio_paths = [Path(p) for p in frame["audio_path"].tolist()]
    pred_labels: list[str] = []
    probabilities: list[np.ndarray] = []

    model.eval()
    model.to(device)

    with torch.inference_mode():
        for chunk in batched(audio_paths, batch_size):
            batch_audio = [
                load_audio(
                    audio_path=path,
                    target_sr=feature_extractor.sampling_rate,
                    max_duration=max_duration,
                )
                for path in chunk
            ]
            features = feature_extractor(
                batch_audio,
                sampling_rate=feature_extractor.sampling_rate,
                truncation=True,
                max_length=int(feature_extractor.sampling_rate * max_duration),
                padding=True,
                return_tensors="pt",
            )
            features = {key: value.to(device) for key, value in features.items()}
            logits = model(**features).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred_ids = logits.argmax(dim=-1).cpu().tolist()

            probabilities.extend(probs)
            pred_labels.extend(model.config.id2label[int(idx)] for idx in pred_ids)

    return pred_labels, np.vstack(probabilities)


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", labels=TARGET_LABELS, zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", labels=TARGET_LABELS, zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", labels=TARGET_LABELS, zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", labels=TARGET_LABELS, zero_division=0)
        ),
    }


def save_confusion_matrix(out_path: Path, y_true: list[str], y_pred: list[str]) -> None:
    extra_pred_labels = sorted(label for label in set(y_pred) if label not in TARGET_LABELS)
    labels = [*TARGET_LABELS, *extra_pred_labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Whisper SER Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(Path(__file__).resolve())
    out_dir = (repo_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    eval_df = build_eval_frame(repo_root=repo_root)

    train_df, test_df = train_test_split(
        eval_df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=eval_df["eval_label"],
    )
    test_df = test_df.reset_index(drop=True)

    if args.limit is not None:
        test_df = test_df.iloc[: args.limit].reset_index(drop=True)

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_id, do_normalize=True)
    model = AutoModelForAudioClassification.from_pretrained(args.model_id)

    native_pred_labels, probability_array = run_inference(
        model=model,
        feature_extractor=feature_extractor,
        frame=test_df,
        device=device,
        batch_size=args.batch_size,
        max_duration=args.max_duration,
    )

    mapped_pred_labels = [MODEL_TO_IEMOCAP_LABEL.get(label, "unmapped") for label in native_pred_labels]
    y_true = test_df["eval_label"].tolist()
    y_pred = mapped_pred_labels

    metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=TARGET_LABELS,
        output_dict=True,
        zero_division=0,
    )

    probability_cols = {
        f"prob_{MODEL_TO_IEMOCAP_LABEL.get(model.config.id2label[int(idx)], model.config.id2label[int(idx)])}": probability_array[
            :, idx
        ]
        for idx in range(probability_array.shape[1])
    }
    predictions_df = test_df[
        ["session", "method", "gender", "emotion", "eval_label", "path", "audio_path"]
    ].copy()
    predictions_df["pred_model_label_native"] = native_pred_labels
    predictions_df["pred_eval_label"] = mapped_pred_labels
    for col_name, col_values in probability_cols.items():
        predictions_df[col_name] = col_values

    results_df = pd.DataFrame(
        [
            {
                "stage": "test",
                "model": "whisper_large_v3_ser",
                **metrics,
                "test_rows": int(len(test_df)),
            }
        ]
    )

    results_df.to_csv(out_dir / "results.csv", index=False)
    predictions_df.to_csv(out_dir / "test_predictions.csv", index=False)
    pd.DataFrame(report).transpose().to_csv(out_dir / "classification_report.csv")
    save_confusion_matrix(out_dir / "confusion_matrix.png", y_true=y_true, y_pred=y_pred)

    run_metadata = {
        "model_id": args.model_id,
        "task": "iemocap_whisper_ser_evaluation",
        "label_policy": {
            "iemocap_to_eval_label": {
                **IEMOCAP_TO_EVAL_LABEL,
            },
            "model_to_iemocap_label": MODEL_TO_IEMOCAP_LABEL,
            "target_labels": TARGET_LABELS,
            "note": (
                "This evaluator matches the repo's feature-selection workflow style "
                "with a stratified 80/20 split, using only the clean label overlaps "
                "between IEMOCAP and the upstream Whisper checkpoint. "
                "Excited, frustrated, other, xxx, and disgust are excluded."
            ),
        },
        "split": {
            "test_size": args.test_size,
            "random_state": args.random_state,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
        "inference": {
            "device": str(device),
            "batch_size": args.batch_size,
            "max_duration": args.max_duration,
            "limit": args.limit,
        },
        "metrics": metrics,
        "classification_report": report,
        "artifacts": {
            "results_csv": str(out_dir / "results.csv"),
            "predictions_csv": str(out_dir / "test_predictions.csv"),
            "classification_report_csv": str(out_dir / "classification_report.csv"),
            "confusion_matrix_png": str(out_dir / "confusion_matrix.png"),
        },
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(to_builtin(run_metadata), indent=2), encoding="utf-8")

    print(f"Repository root: {repo_root}")
    print(f"Output dir: {out_dir}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
