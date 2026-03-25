from __future__ import annotations

import argparse
import inspect
import json
import os
import socket
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
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
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

MODEL_ID = "openai/whisper-large-v3"
BASE_TARGET_LABELS = ["fru", "neu", "ang", "sad", "exc", "hap"]
DEFAULT_OUTPUT_DIR = "whisper/finetuning/artifacts"
DEFAULT_TEST_SIZE = 0.20
DEFAULT_VAL_SIZE = 0.10
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_DURATION = 30.0
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_TRAIN_BATCH_SIZE = 2
DEFAULT_EVAL_BATCH_SIZE = 2
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_NUM_TRAIN_EPOCHS = 5.0
DEFAULT_WARMUP_RATIO = 0.10
DEFAULT_LOGGING_STEPS = 10
DEFAULT_EVAL_STEPS = 100
DEFAULT_SAVE_TOTAL_LIMIT = 2

BASE_IEMOCAP_TO_LABEL = {
    "fru": "fru",
    "neu": "neu",
    "ang": "ang",
    "sad": "sad",
    "exc": "exc",
    "hap": "hap",
}


@dataclass
class VariantArtifacts:
    variant: str
    output_dir: str
    report_dir: str
    figures_dir: str
    tensorboard_dir: str


def parse_report_to(value: str) -> list[str]:
    reporters = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not reporters:
        raise argparse.ArgumentTypeError("Expected at least one reporting backend.")

    allowed = {"tensorboard", "wandb", "none"}
    invalid = [item for item in reporters if item not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported report backend(s): {', '.join(invalid)}")
    if "none" in reporters and len(reporters) > 1:
        raise argparse.ArgumentTypeError("'none' cannot be combined with other reporting backends.")
    return reporters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finetune Whisper for speech emotion recognition on IEMOCAP with "
            "step-wise logging/evaluation/checkpointing, Slurm-friendly outputs, "
            "and saved test visualizations."
        )
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional name for the run directory. Defaults to a timestamped name.",
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument(
        "--max-samples-per-label",
        type=int,
        default=None,
        help="Optional cap applied before splitting so each label contributes at most N rows.",
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument("--num-train-epochs", type=float, default=DEFAULT_NUM_TRAIN_EPOCHS)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Optional explicit warmup steps. Overrides --warmup-ratio when set.",
    )
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS)
    parser.add_argument(
        "--schedule-mode",
        choices=["steps", "epoch"],
        default="steps",
        help="Whether eval/save run on step intervals or at each epoch boundary.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=DEFAULT_EVAL_STEPS,
        help="Only used when --schedule-mode=steps.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Only used when --schedule-mode=steps. Defaults to eval_steps.",
    )
    parser.add_argument("--save-total-limit", type=int, default=DEFAULT_SAVE_TOTAL_LIMIT)
    parser.add_argument(
        "--precision",
        choices=["auto", "fp32", "fp16", "bf16"],
        default="auto",
        help="Precision mode. auto selects fp16 on CUDA and fp32 otherwise.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Force the primary training device.",
    )
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument(
        "--include-xxx",
        action="store_true",
        help="Include the optional xxx class when running a single variant.",
    )
    parser.add_argument(
        "--run-both-xxx-variants",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run both with_xxx and without_xxx variants.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint path to resume from, or 'latest' to auto-discover.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,
        help="Worker count for PyTorch data loaders. Start low on shared storage.",
    )
    parser.add_argument(
        "--report-to",
        type=parse_report_to,
        default=["tensorboard"],
        help="Comma-separated reporting backends: tensorboard, wandb, or none.",
    )
    parser.add_argument(
        "--save-figures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save training curves and test confusion matrices as PNGs.",
    )
    return parser.parse_args()


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root from the current path.")


def normalize_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_precision_flags(precision: str, device: torch.device) -> tuple[bool, bool, str]:
    if precision == "fp32":
        return False, False, "fp32"
    if precision == "fp16":
        if device.type != "cuda":
            raise RuntimeError("fp16 was requested but CUDA is not available.")
        return True, False, "fp16"
    if precision == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested but is not supported on this CUDA device.")
        return False, True, "bf16"
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return False, True, "bf16"
        return True, False, "fp16"
    return False, False, "fp32"


def target_labels_for_variant(include_xxx: bool) -> list[str]:
    return [*BASE_TARGET_LABELS, "xxx"] if include_xxx else list(BASE_TARGET_LABELS)


def label_map_for_variant(include_xxx: bool) -> dict[str, str]:
    label_map = dict(BASE_IEMOCAP_TO_LABEL)
    if include_xxx:
        label_map["xxx"] = "xxx"
    return label_map


def variant_name(include_xxx: bool) -> str:
    return "with_xxx" if include_xxx else "without_xxx"


def build_training_frame(repo_root: Path, include_xxx: bool) -> pd.DataFrame:
    metadata_path = repo_root / "datasets" / "IEMOCAP" / "iemocap_full_dataset.csv"
    audio_root = repo_root / "datasets" / "IEMOCAP"
    if not metadata_path.exists():
        raise FileNotFoundError(f"IEMOCAP metadata not found: {metadata_path}")

    raw_df = pd.read_csv(metadata_path)
    raw_df["emotion"] = normalize_series(raw_df["emotion"])
    raw_df["method"] = normalize_series(raw_df["method"])
    raw_df["gender"] = raw_df["gender"].astype(str).str.strip().str.upper()
    raw_df["path"] = raw_df["path"].astype(str).str.replace("\\", "/", regex=False).str.strip()

    label_map = label_map_for_variant(include_xxx)
    df = raw_df[raw_df["emotion"].isin(label_map)].copy()
    if include_xxx:
        df = df[(df["emotion"] == "xxx") | (df["agreement"] > 0)].copy()
    else:
        df = df[df["agreement"] > 0].copy()

    df["label"] = df["emotion"].map(label_map)
    df["audio_path"] = df["path"].map(lambda rel: str((audio_root / rel).resolve()))
    df["audio_exists"] = df["audio_path"].map(lambda p: Path(p).exists())
    df = df[df["audio_exists"]].reset_index(drop=True)
    label_to_id = {label: idx for idx, label in enumerate(target_labels_for_variant(include_xxx))}
    df["label_id"] = df["label"].map(label_to_id)

    if df.empty:
        raise RuntimeError(f"No rows remained for variant={variant_name(include_xxx)} after filtering.")

    return df


def cap_samples_per_label(df: pd.DataFrame, max_samples_per_label: int | None, random_state: int) -> pd.DataFrame:
    if max_samples_per_label is None:
        return df

    shuffled_df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    capped_df = (
        shuffled_df.groupby("label", group_keys=False)
        .head(max_samples_per_label)
        .reset_index(drop=True)
    )
    return capped_df


def make_splits(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_val_df["label"],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def load_audio(audio_path: Path, target_sr: int, max_duration: float) -> np.ndarray:
    audio, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    max_length = int(target_sr * max_duration)
    if len(audio) >= max_length:
        return audio[:max_length].astype(np.float32, copy=False)

    padded = np.zeros(max_length, dtype=np.float32)
    padded[: len(audio)] = audio.astype(np.float32, copy=False)
    return padded


class IemocapWhisperDataset(torch.utils.data.Dataset):
    def __init__(self, frame: pd.DataFrame, feature_extractor: Any, max_duration: float):
        self.frame = frame.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.max_duration = max_duration

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[idx]
        audio = load_audio(
            Path(row["audio_path"]),
            target_sr=self.feature_extractor.sampling_rate,
            max_duration=self.max_duration,
        )
        features = self.feature_extractor(
            audio,
            sampling_rate=self.feature_extractor.sampling_rate,
            truncation=True,
            max_length=int(self.feature_extractor.sampling_rate * self.max_duration),
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in features.items()}
        item["labels"] = torch.tensor(int(row["label_id"]), dtype=torch.long)
        return item


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_weighted": float(precision_score(labels, preds, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(labels, preds, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(to_builtin(payload), indent=2), encoding="utf-8")


def collect_gpu_memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device_index)
    reserved = torch.cuda.memory_reserved(device_index)
    max_allocated = torch.cuda.max_memory_allocated(device_index)
    max_reserved = torch.cuda.max_memory_reserved(device_index)
    bytes_per_gib = float(1024**3)
    return {
        "gpu_memory_allocated_gib": allocated / bytes_per_gib,
        "gpu_memory_reserved_gib": reserved / bytes_per_gib,
        "gpu_memory_peak_allocated_gib": max_allocated / bytes_per_gib,
        "gpu_memory_peak_reserved_gib": max_reserved / bytes_per_gib,
    }


class GpuMemoryLoggingCallback(TrainerCallback):
    def __init__(self, device: torch.device):
        self.device = device

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        if logs is None:
            return control
        logs.update(collect_gpu_memory_stats(self.device))
        return control


def save_confusion_matrix_figure(
    path: Path,
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    value_format: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=value_format, cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_history_artifacts(history_df: pd.DataFrame, figures_dir: Path, save_figures: bool) -> None:
    history_df.to_csv(figures_dir / "trainer_log_history.csv", index=False)
    save_json(figures_dir / "trainer_log_history.json", history_df.to_dict(orient="records"))

    if not save_figures or history_df.empty:
        return

    train_df = history_df.dropna(subset=["loss", "step"]).copy()
    eval_df = history_df.dropna(subset=["eval_loss", "step"]).copy()

    if not train_df.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(train_df["step"], train_df["loss"], label="train_loss")
        ax.set_title("Training Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / "train_loss_curve.png", dpi=160)
        plt.close(fig)

    if not eval_df.empty:
        metrics_to_plot = [metric for metric in ["eval_loss", "eval_f1_macro", "eval_accuracy"] if metric in eval_df]
        for metric in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(eval_df["step"], eval_df[metric], marker="o", label=metric)
            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlabel("Step")
            ax.set_ylabel(metric)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(figures_dir / f"{metric}_curve.png", dpi=160)
            plt.close(fig)


def discover_resume_checkpoint(variant_dir: Path, requested: str | None) -> str | None:
    if requested is None:
        return None
    if requested != "latest":
        return requested

    checkpoints = sorted(
        [path for path in variant_dir.glob("checkpoint-*") if path.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if checkpoints:
        return str(checkpoints[-1])
    return None


def build_training_args(
    *,
    variant_dir: Path,
    tensorboard_dir: Path,
    args: argparse.Namespace,
    fp16: bool,
    bf16: bool,
) -> TrainingArguments:
    strategy = args.schedule_mode
    eval_steps = args.eval_steps if strategy == "steps" else None
    save_steps = args.save_steps if strategy == "steps" else None
    if strategy == "steps" and save_steps is None:
        save_steps = eval_steps

    kwargs: dict[str, Any] = {
        "output_dir": str(variant_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": args.logging_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "report_to": [] if args.report_to == ["none"] else args.report_to,
        "fp16": fp16,
        "bf16": bf16,
        "remove_unused_columns": False,
        "save_total_limit": args.save_total_limit,
        "dataloader_num_workers": args.dataloader_num_workers,
    }

    if args.warmup_steps is not None:
        kwargs["warmup_steps"] = args.warmup_steps
    else:
        kwargs["warmup_ratio"] = args.warmup_ratio

    if "tensorboard" in args.report_to:
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(tensorboard_dir)

    signature = inspect.signature(TrainingArguments.__init__)
    params = signature.parameters
    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = strategy
    elif "eval_strategy" in params:
        kwargs["eval_strategy"] = strategy

    if "logging_strategy" in params:
        kwargs["logging_strategy"] = "steps"

    if "save_strategy" in params:
        kwargs["save_strategy"] = strategy

    if "save_safetensors" in params:
        kwargs["save_safetensors"] = True

    if strategy == "steps":
        if "eval_steps" in params:
            kwargs["eval_steps"] = eval_steps
        if "save_steps" in params:
            kwargs["save_steps"] = save_steps

    return TrainingArguments(**kwargs)


def save_test_artifacts(
    *,
    trainer: Trainer,
    feature_extractor: Any,
    test_df: pd.DataFrame,
    test_dataset: IemocapWhisperDataset,
    labels: list[str],
    variant_dir: Path,
    save_figures: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    reports_dir = variant_dir / "reports"
    figures_dir = variant_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    prediction_output = trainer.predict(test_dataset, metric_key_prefix="test")
    pred_ids = np.argmax(prediction_output.predictions, axis=-1)
    probabilities = torch.softmax(torch.tensor(prediction_output.predictions), dim=-1).numpy()

    id_to_label = {idx: label for idx, label in enumerate(labels)}
    y_true = [id_to_label[int(label_id)] for label_id in test_df["label_id"].tolist()]
    y_pred = [id_to_label[int(pred_id)] for pred_id in pred_ids.tolist()]

    prediction_df = test_df.copy()
    prediction_df["true_label"] = y_true
    prediction_df["pred_label"] = y_pred
    prediction_df["pred_correct"] = prediction_df["true_label"] == prediction_df["pred_label"]
    prediction_df["pred_confidence"] = probabilities.max(axis=1)
    for idx, label in enumerate(labels):
        prediction_df[f"prob_{label}"] = probabilities[:, idx]
    prediction_df.to_csv(reports_dir / "test_predictions.csv", index=False)

    metric_table_df = pd.DataFrame(
        [{"metric": key, "value": float(value)} for key, value in test_metrics.items() if isinstance(value, (int, float))]
    )
    metric_table_df.to_csv(reports_dir / "test_metrics.csv", index=False)
    save_json(reports_dir / "test_metrics.json", test_metrics)

    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "label"})
    report_df.to_csv(reports_dir / "classification_report.csv", index=False)
    save_json(reports_dir / "classification_report.json", report_dict)

    counts_cm = confusion_matrix(y_true, y_pred, labels=labels)
    normalized_cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    counts_df = pd.DataFrame(counts_cm, index=labels, columns=labels)
    normalized_df = pd.DataFrame(normalized_cm, index=labels, columns=labels)
    counts_df.to_csv(reports_dir / "confusion_matrix_counts.csv")
    normalized_df.to_csv(reports_dir / "confusion_matrix_normalized.csv")

    if save_figures:
        save_confusion_matrix_figure(
            figures_dir / "confusion_matrix_counts.png",
            counts_cm,
            labels,
            "Test Confusion Matrix (Counts)",
            "d",
        )
        save_confusion_matrix_figure(
            figures_dir / "confusion_matrix_normalized.png",
            normalized_cm,
            labels,
            "Test Confusion Matrix (Normalized)",
            ".2f",
        )

    preview_columns = [
        "session",
        "file",
        "emotion",
        "label",
        "true_label",
        "pred_label",
        "pred_correct",
        "pred_confidence",
    ]
    existing_preview_columns = [column for column in preview_columns if column in prediction_df.columns]
    preview_path = reports_dir / "test_prediction_preview.csv"
    prediction_df[existing_preview_columns].head(50).to_csv(preview_path, index=False)

    artifacts = VariantArtifacts(
        variant=variant_dir.name,
        output_dir=str(variant_dir),
        report_dir=str(reports_dir),
        figures_dir=str(figures_dir),
        tensorboard_dir=str(variant_dir / "runs"),
    )
    feature_extractor.save_pretrained(variant_dir)
    return test_metrics, asdict(artifacts)


def summarize_environment(device: torch.device) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "host": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "python_executable": sys.executable,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if torch.cuda.is_available():
        summary["cuda_device_count"] = torch.cuda.device_count()
        summary["cuda_device_name"] = torch.cuda.get_device_name(0)
        total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        summary["cuda_total_memory_gib"] = total_memory_bytes / float(1024**3)
    return summary


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_stub = args.model_id.split("/")[-1].replace("-", "_")
    return f"{timestamp}_{model_stub}_iemocap_finetune"


def train_variant(
    *,
    include_xxx: bool,
    args: argparse.Namespace,
    repo_root: Path,
    run_dir: Path,
    feature_extractor: Any,
    device: torch.device,
    fp16: bool,
    bf16: bool,
) -> dict[str, Any]:
    name = variant_name(include_xxx)
    labels = target_labels_for_variant(include_xxx)
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    variant_dir = run_dir / name
    tensorboard_dir = variant_dir / "runs"
    variant_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    frame = build_training_frame(repo_root=repo_root, include_xxx=include_xxx)
    frame = cap_samples_per_label(
        frame,
        max_samples_per_label=args.max_samples_per_label,
        random_state=args.random_state,
    )
    train_df, val_df, test_df = make_splits(
        frame,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    if args.max_train_samples is not None:
        train_df = train_df.iloc[: args.max_train_samples].reset_index(drop=True)
    if args.max_eval_samples is not None:
        val_df = val_df.iloc[: args.max_eval_samples].reset_index(drop=True)
        test_df = test_df.iloc[: args.max_eval_samples].reset_index(drop=True)

    split_summary_df = pd.DataFrame(
        [
            {"split": "train", "rows": len(train_df)},
            {"split": "val", "rows": len(val_df)},
            {"split": "test", "rows": len(test_df)},
        ]
    )
    split_summary_df.to_csv(variant_dir / "split_summary.csv", index=False)

    label_distribution_df = pd.DataFrame(
        [
            {"split": split_name, "label": label, "count": int(count)}
            for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]
            for label, count in split_df["label"].value_counts().sort_index().items()
        ]
    )
    label_distribution_df.to_csv(variant_dir / "label_distribution.csv", index=False)

    model = AutoModelForAudioClassification.from_pretrained(
        args.model_id,
        num_labels=len(labels),
        label2id=label_to_id,
        id2label=id_to_label,
        ignore_mismatched_sizes=True,
    )
    if not fp16 and not bf16:
        model = model.float()
    if args.freeze_encoder and hasattr(model, "freeze_encoder"):
        model.freeze_encoder()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_dataset = IemocapWhisperDataset(train_df, feature_extractor=feature_extractor, max_duration=args.max_duration)
    val_dataset = IemocapWhisperDataset(val_df, feature_extractor=feature_extractor, max_duration=args.max_duration)
    test_dataset = IemocapWhisperDataset(test_df, feature_extractor=feature_extractor, max_duration=args.max_duration)

    training_args = build_training_args(
        variant_dir=variant_dir,
        tensorboard_dir=tensorboard_dir,
        args=args,
        fp16=fp16,
        bf16=bf16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[GpuMemoryLoggingCallback(device)],
    )

    resume_checkpoint = discover_resume_checkpoint(variant_dir=variant_dir, requested=args.resume_from_checkpoint)
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    val_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="val")
    trainer.save_model()

    history_df = pd.DataFrame(trainer.state.log_history)
    figures_dir = variant_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_history_artifacts(history_df=history_df, figures_dir=figures_dir, save_figures=args.save_figures)

    test_metrics, artifact_paths = save_test_artifacts(
        trainer=trainer,
        feature_extractor=feature_extractor,
        test_df=test_df,
        test_dataset=test_dataset,
        labels=labels,
        variant_dir=variant_dir,
        save_figures=args.save_figures,
    )
    gpu_memory_summary = collect_gpu_memory_stats(device)

    metadata = {
        "variant": name,
        "include_xxx": include_xxx,
        "labels": labels,
        "train_metrics": train_result.metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "resume_from_checkpoint": resume_checkpoint,
        "training_args": training_args.to_dict(),
        "artifacts": artifact_paths,
        "gpu_memory_summary": gpu_memory_summary,
    }
    save_json(variant_dir / "run_metadata.json", metadata)

    summary: dict[str, Any] = {
        "variant": name,
        "include_xxx": include_xxx,
        "target_labels": ", ".join(labels),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "output_dir": str(variant_dir),
        "report_dir": artifact_paths["report_dir"],
        "figures_dir": artifact_paths["figures_dir"],
        "tensorboard_dir": artifact_paths["tensorboard_dir"],
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
    }
    summary.update(gpu_memory_summary)
    for source in (train_result.metrics, val_metrics, test_metrics):
        for key, value in source.items():
            if isinstance(value, (int, float)):
                summary[key] = float(value)
    return summary


def save_run_overview(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    environment_summary: dict[str, Any],
    variant_results: list[dict[str, Any]],
) -> None:
    run_summary_df = pd.DataFrame(variant_results)
    run_summary_df.to_csv(run_dir / "variant_comparison.csv", index=False)
    save_json(run_dir / "variant_comparison.json", variant_results)
    save_json(run_dir / "environment_summary.json", environment_summary)
    save_json(run_dir / "run_config.json", vars(args))

    if len(run_summary_df) > 1 and args.save_figures:
        metric_columns = [column for column in ["test_accuracy", "test_f1_macro", "test_f1_weighted"] if column in run_summary_df]
        for metric in metric_columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(run_summary_df["variant"], run_summary_df[metric])
            ax.set_title(metric.replace("_", " ").title())
            ax.set_ylabel(metric)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(run_dir / f"{metric}_comparison.png", dpi=160)
            plt.close(fig)

    latest_path = run_dir.parent / "latest_run.txt"
    latest_path.write_text(f"{run_dir}\n", encoding="utf-8")


def print_start_banner(run_dir: Path, environment_summary: dict[str, Any], args: argparse.Namespace) -> None:
    print("=== Whisper SER Finetuning ===")
    print(f"Run directory: {run_dir}")
    print(f"Host: {environment_summary['host']}")
    print(f"Device: {environment_summary['device']}")
    if environment_summary.get("cuda_device_name"):
        print(f"CUDA device: {environment_summary['cuda_device_name']}")
    print(f"Schedule mode: {args.schedule_mode}")
    print(f"Base target labels: {', '.join(BASE_TARGET_LABELS)}")
    if args.schedule_mode == "steps":
        print(f"Logging steps: {args.logging_steps}")
        print(f"Eval steps: {args.eval_steps}")
        print(f"Save steps: {args.save_steps or args.eval_steps}")
    print(f"Reporting backends: {', '.join(args.report_to)}")


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)
    repo_root = find_repo_root(Path(__file__).resolve())
    run_root = (repo_root / args.output_dir).resolve()
    run_dir = run_root / build_run_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    fp16, bf16, effective_precision = resolve_precision_flags(args.precision, device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_id, do_normalize=True)

    environment_summary = summarize_environment(device)
    environment_summary["sampling_rate"] = feature_extractor.sampling_rate
    environment_summary["effective_precision"] = effective_precision
    environment_summary["base_target_labels"] = list(BASE_TARGET_LABELS)
    print_start_banner(run_dir=run_dir, environment_summary=environment_summary, args=args)
    if "wandb" in args.report_to:
        try:
            import wandb  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "wandb reporting was requested but the package is not installed in the active environment."
            ) from exc

    variants_to_run = [True, False] if args.run_both_xxx_variants else [args.include_xxx]
    variant_results = [
        train_variant(
            include_xxx=include_xxx,
            args=args,
            repo_root=repo_root,
            run_dir=run_dir,
            feature_extractor=feature_extractor,
            device=device,
            fp16=fp16,
            bf16=bf16,
        )
        for include_xxx in variants_to_run
    ]

    save_run_overview(
        run_dir=run_dir,
        args=args,
        environment_summary=environment_summary,
        variant_results=variant_results,
    )

    print("=== Completed ===")
    print(pd.DataFrame(variant_results).to_string(index=False))
    print(f"Variant comparison CSV: {run_dir / 'variant_comparison.csv'}")
    print(f"Latest run marker: {run_root / 'latest_run.txt'}")


if __name__ == "__main__":
    main()
