from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import json
import random
import socket
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from Emotion2Vec import EMOTION_LABELS, ID_TO_LABEL
from Emotion2Vec.data.dataset import Emotion2VecEmbeddingDataset, collate_emotion2vec_batch
from Emotion2Vec.evaluation.metrics import (
    build_classification_report_frame,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_normalized_confusion_matrix,
    plot_confusion_matrix,
)
from Emotion2Vec.models.downstream_classifier import Emotion2VecDownstreamClassifier
from utils.wandb_multi import MultiWandbRun, init_multi_wandb_run

DEFAULT_CONFIG_PATH = "configs/downstream_config.yaml"
DEFAULT_OUTPUT_ROOT = "Emotion2Vec/artifacts"


@dataclass
class EpochResult:
    split: str
    loss: float
    metrics: dict[str, float]
    y_true: list[int]
    y_pred: list[int]
    sample_ids: list[str]
    embedding_paths: list[str]
    lengths: list[int]


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    bootstrap_args, remaining = bootstrap.parse_known_args()

    config_defaults = load_yaml_config(bootstrap_args.config)
    parser = build_parser(config_defaults, bootstrap_args.config)
    return parser.parse_args(remaining)


def build_parser(config_defaults: dict[str, Any], config_path: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a non-sequential downstream SER classifier on frozen emotion2vec embeddings. "
            "Inputs are pre-extracted frame embeddings, not raw audio."
        )
    )

    parser.add_argument("--config", default=config_path)
    parser.add_argument("--train-csv", default=config_defaults.get("train_csv"))
    parser.add_argument("--val-csv", default=config_defaults.get("val_csv"))
    parser.add_argument("--test-csv", default=config_defaults.get("test_csv"))
    parser.add_argument("--output-dir", default=config_defaults.get("output_dir", DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-name", default=config_defaults.get("run_name"))
    parser.add_argument("--device", default=config_defaults.get("device", "auto"))
    parser.add_argument("--seed", type=int, default=int(config_defaults.get("seed", 42)))
    parser.add_argument(
        "--resume-from-checkpoint",
        default=config_defaults.get("resume_from_checkpoint"),
        help="Resume from 'last', 'best', or an explicit checkpoint path.",
    )

    parser.add_argument("--label-col", default=config_defaults.get("label_col", "label"))
    parser.add_argument("--embedding-path-col", default=config_defaults.get("embedding_path_col", "embedding_path"))
    parser.add_argument("--audio-path-col", default=config_defaults.get("audio_path_col"))
    parser.add_argument("--sample-id-col", default=config_defaults.get("sample_id_col"))
    parser.add_argument("--embeddings-root", default=config_defaults.get("embeddings_root"))
    parser.add_argument("--embedding-suffix", default=config_defaults.get("embedding_suffix", ".pt"))
    parser.add_argument("--embedding-dim", type=int, default=int(config_defaults.get("embedding_dim", 768)))
    parser.add_argument("--hidden-dim", type=int, default=int(config_defaults.get("hidden_dim", 256)))

    parser.add_argument("--epochs", type=int, default=int(config_defaults.get("epochs", 30)))
    parser.add_argument("--batch-size", type=int, default=int(config_defaults.get("batch_size", 32)))
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=int(config_defaults.get("eval_batch_size", config_defaults.get("batch_size", 32))),
    )
    parser.add_argument("--learning-rate", type=float, default=float(config_defaults.get("learning_rate", 1e-3)))
    parser.add_argument("--weight-decay", type=float, default=float(config_defaults.get("weight_decay", 1e-4)))
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default=config_defaults.get("optimizer", "adamw"))
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "reduce_on_plateau"],
        default=config_defaults.get("lr_scheduler", "reduce_on_plateau"),
        help="Learning-rate scheduler strategy.",
    )
    scheduler_patience_default = config_defaults.get("lr_scheduler_patience", 2)
    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=int(scheduler_patience_default),
        help="Scheduler patience in epochs before reducing LR.",
    )
    parser.add_argument(
        "--lr-scheduler-factor",
        type=float,
        default=float(config_defaults.get("lr_scheduler_factor", 0.5)),
        help="Multiplicative factor applied when the scheduler reduces LR.",
    )
    parser.add_argument(
        "--lr-scheduler-min-lr",
        type=float,
        default=float(config_defaults.get("lr_scheduler_min_lr", 1e-6)),
        help="Lower bound for scheduler-controlled learning rates.",
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=float(config_defaults.get("gradient_clip_norm", 1.0)),
    )
    parser.add_argument("--num-workers", type=int, default=int(config_defaults.get("num_workers", 0)))
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=int(config_defaults.get("log_every_steps", 1)),
        help="Log batch-level training metrics to W&B every N optimizer steps.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=bool(config_defaults.get("pin_memory", True)),
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=config_defaults.get("max_train_samples"),
        help="Optional cap for train rows, useful for smoke tests.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=config_defaults.get("max_val_samples"),
        help="Optional cap for validation rows, useful for smoke tests.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=config_defaults.get("max_test_samples"),
        help="Optional cap for test rows, useful for smoke tests.",
    )
    early_stopping_patience_default = config_defaults.get("early_stopping_patience")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=int(early_stopping_patience_default) if early_stopping_patience_default is not None else None,
        help="Stop after N consecutive epochs without a validation macro-F1 improvement.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=float(config_defaults.get("early_stopping_min_delta", 0.0)),
        help="Minimum validation macro-F1 improvement required to reset early stopping patience.",
    )

    parser.add_argument("--report-to", choices=["wandb", "none"], default=config_defaults.get("report_to", "none"))
    parser.add_argument("--wandb-project", default=config_defaults.get("wandb_project", "emotion2vec-downstream-ser"))
    parser.add_argument("--wandb-entity", default=config_defaults.get("wandb_entity"))
    parser.add_argument("--wandb-group", default=config_defaults.get("wandb_group"))
    parser.add_argument("--wandb-tags", default=config_defaults.get("wandb_tags", "emotion2vec,downstream,ser"))
    parser.add_argument("--wandb-mode", default=config_defaults.get("wandb_mode", "online"))
    parser.add_argument(
        "--wandb-watch-model",
        action=argparse.BooleanOptionalAction,
        default=bool(config_defaults.get("wandb_watch_model", False)),
        help=(
            "Opt in to W&B model watching. Disabled by default because watch() installs "
            "autograd hooks that can fail inside long-running cluster jobs."
        ),
    )

    parser.add_argument(
        "--save-figures",
        action=argparse.BooleanOptionalAction,
        default=bool(config_defaults.get("save_figures", True)),
    )
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=bool(config_defaults.get("save_predictions", True)),
    )

    return parser


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser()
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in config file {path}, got {type(data).__name__}.")
    return data


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device(device_arg)


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root from the current path.")


def resolve_repo_path(path_value: str | Path | None, repo_root: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_required_args(args: argparse.Namespace) -> None:
    required = {"train_csv": args.train_csv, "val_csv": args.val_csv, "test_csv": args.test_csv}
    missing = [key for key, value in required.items() if not value]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required CLI/config value(s): {joined}")


def make_run_dir(args: argparse.Namespace, repo_root: Path) -> Path:
    output_root = resolve_repo_path(args.output_dir, repo_root)
    if output_root is None:
        raise ValueError("output_dir must be set.")

    run_name = args.run_name or f"emotion2vec_downstream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def maybe_limit_rows(csv_path: Path, limit: int | None) -> Path | pd.DataFrame:
    if limit is None:
        return csv_path
    return pd.read_csv(csv_path).head(limit)


def create_dataloaders(
    args: argparse.Namespace,
    repo_root: Path,
    device: torch.device,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    train_csv = maybe_limit_rows(resolve_repo_path(args.train_csv, repo_root), args.max_train_samples)
    val_csv = maybe_limit_rows(resolve_repo_path(args.val_csv, repo_root), args.max_val_samples)
    test_csv = maybe_limit_rows(resolve_repo_path(args.test_csv, repo_root), args.max_test_samples)

    common_dataset_kwargs = {
        "repo_root": repo_root,
        "label_col": args.label_col,
        "embedding_path_col": args.embedding_path_col,
        "audio_path_col": args.audio_path_col,
        "sample_id_col": args.sample_id_col,
        "embeddings_root": args.embeddings_root,
        "embedding_suffix": args.embedding_suffix,
        "expected_dim": args.embedding_dim,
        "allowed_labels": EMOTION_LABELS,
    }

    train_dataset = Emotion2VecEmbeddingDataset(train_csv, **common_dataset_kwargs)
    val_dataset = Emotion2VecEmbeddingDataset(val_csv, **common_dataset_kwargs)
    test_dataset = Emotion2VecEmbeddingDataset(test_csv, **common_dataset_kwargs)

    pin_memory = bool(args.pin_memory and device.type == "cuda")
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_emotion2vec_batch,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "embeddings": batch["embeddings"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
        "lengths": batch["lengths"],
        "sample_ids": batch["sample_ids"],
        "label_strs": batch["label_strs"],
        "embedding_paths": batch["embedding_paths"],
    }


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> Optimizer:
    parameters = model.parameters()
    if args.optimizer == "adam":
        return Adam(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    return AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)


def build_scheduler(optimizer: Optimizer, args: argparse.Namespace) -> ReduceLROnPlateau | None:
    if args.lr_scheduler == "none":
        return None
    return ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        min_lr=args.lr_scheduler_min_lr,
    )


def resolve_resume_checkpoint(
    *,
    requested: str | None,
    run_dir: Path,
    repo_root: Path,
) -> Path | None:
    if requested is None:
        return None

    normalized = str(requested).strip()
    if not normalized:
        return None

    if normalized == "last":
        checkpoint_path = run_dir / "last_model.pt"
    elif normalized == "best":
        checkpoint_path = run_dir / "best_model.pt"
    else:
        candidate = Path(normalized).expanduser()
        checkpoint_path = candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def move_optimizer_state_to_device(optimizer: Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def load_rows_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    return pd.read_csv(csv_path).to_dict(orient="records")


def write_progress_tables(
    *,
    run_dir: Path,
    history_rows: list[dict[str, Any]],
    train_step_rows: list[dict[str, Any]],
) -> None:
    pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame(train_step_rows).to_csv(run_dir / "train_steps.csv", index=False)


def restore_random_state(checkpoint: dict[str, Any]) -> None:
    python_state = checkpoint.get("python_random_state")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = checkpoint.get("numpy_random_state")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    torch_state = checkpoint.get("torch_random_state")
    if torch_state is not None:
        torch.random.set_rng_state(torch_state)

    cuda_state = checkpoint.get("cuda_random_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def load_training_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def count_labels(dataset: Emotion2VecEmbeddingDataset) -> pd.DataFrame:
    counts = pd.Series([record.label for record in dataset.records], dtype="string").value_counts().sort_index()
    return pd.DataFrame({"label": counts.index.tolist(), "count": counts.astype(int).tolist()})


def collect_frame_lengths(dataset: Emotion2VecEmbeddingDataset) -> list[int]:
    lengths: list[int] = []
    for record in dataset.records:
        tensor = torch.load(record.embedding_path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected a tensor in {record.embedding_path}, got {type(tensor).__name__}.")
        lengths.append(int(tensor.shape[0]))
    return lengths


def plot_label_distribution(label_frames: dict[str, pd.DataFrame], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = list(EMOTION_LABELS)
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (split_name, frame) in enumerate(label_frames.items()):
        counts_by_label = {row["label"]: int(row["count"]) for _, row in frame.iterrows()}
        values = [counts_by_label.get(label, 0) for label in labels]
        ax.bar(x + (idx - 1) * width, values, width=width, label=split_name)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Label Distribution by Split")
    ax.set_xlabel("Emotion Label")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_length_histogram(lengths_by_split: dict[str, list[int]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for split_name, lengths in lengths_by_split.items():
        ax.hist(lengths, bins=30, alpha=0.45, label=split_name)
    ax.set_title("Embedding Sequence Lengths")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Samples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_per_class_f1(report_frame: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    frame = report_frame[report_frame["label"].isin(EMOTION_LABELS)].copy()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(frame["label"], frame["f1-score"].astype(float))
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Per-Class Test F1")
    ax.set_xlabel("Emotion Label")
    ax.set_ylabel("F1")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_per_class_metrics(report_frame: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    frame = report_frame[report_frame["label"].isin(EMOTION_LABELS)].copy()
    labels = frame["label"].tolist()
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, frame["precision"].astype(float), width=width, label="precision")
    ax.bar(x, frame["recall"].astype(float), width=width, label="recall")
    ax.bar(x + width, frame["f1-score"].astype(float), width=width, label="f1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Per-Class Test Metrics")
    ax.set_xlabel("Emotion Label")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_true_vs_pred_distribution(result: EpochResult, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    true_counts = pd.Series([ID_TO_LABEL[int(label)] for label in result.y_true], dtype="string").value_counts()
    pred_counts = pd.Series([ID_TO_LABEL[int(label)] for label in result.y_pred], dtype="string").value_counts()
    labels = list(EMOTION_LABELS)
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, [int(true_counts.get(label, 0)) for label in labels], width=width, label="true")
    ax.bar(x + width / 2, [int(pred_counts.get(label, 0)) for label in labels], width=width, label="predicted")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Test Label Distribution: True vs Predicted")
    ax.set_xlabel("Emotion Label")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def melt_confusion_matrix(matrix: np.ndarray, label_names: tuple[str, ...], *, normalized: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    value_key = "normalized_value" if normalized else "count"
    for true_idx, true_label in enumerate(label_names):
        for pred_idx, pred_label in enumerate(label_names):
            rows.append(
                {
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    value_key: float(matrix[true_idx, pred_idx]) if normalized else int(matrix[true_idx, pred_idx]),
                }
            )
    return pd.DataFrame(rows)


def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None,
    gradient_clip_norm: float,
    split: str,
    wandb_run: MultiWandbRun | None = None,
    log_every_steps: int = 1,
    starting_global_step: int = 0,
    epoch_index: int | None = None,
) -> tuple[EpochResult, int, list[dict[str, Any]]]:
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_examples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    sample_ids: list[str] = []
    embedding_paths: list[str] = []
    lengths: list[int] = []
    global_step = starting_global_step
    step_rows: list[dict[str, Any]] = []

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device=device)
        logits = model(batch["embeddings"], attention_mask=batch["attention_mask"])
        loss = criterion(logits, batch["labels"])

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if gradient_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            global_step += 1

        batch_size = int(batch["labels"].shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size

        predictions = logits.argmax(dim=-1)
        batch_accuracy = float((predictions == batch["labels"]).float().mean().item())
        lengths.extend(batch["lengths"].tolist())
        y_true.extend(batch["labels"].detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())
        sample_ids.extend(batch["sample_ids"])
        embedding_paths.extend(batch["embedding_paths"])

        if is_training:
            lr = float(optimizer.param_groups[0]["lr"])
            step_row = {
                "global_step": global_step,
                "epoch": epoch_index if epoch_index is not None else split,
                "batch_loss": float(loss.detach().item()),
                "batch_accuracy": batch_accuracy,
                "batch_size": batch_size,
                "learning_rate": lr,
            }
            step_rows.append(step_row)
            if wandb_run is not None and global_step % max(log_every_steps, 1) == 0:
                wandb_run.log(
                    {
                        "train/step_loss": step_row["batch_loss"],
                        "train/step_accuracy": step_row["batch_accuracy"],
                        "train/batch_size": batch_size,
                        "train/learning_rate": lr,
                    },
                    step=global_step,
                )

    average_loss = total_loss / max(total_examples, 1)
    metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred, label_names=EMOTION_LABELS)
    return (
        EpochResult(
            split=split,
            loss=average_loss,
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sample_ids=sample_ids,
            embedding_paths=embedding_paths,
            lengths=lengths,
        ),
        global_step,
        step_rows,
    )


def evaluate_split(
    *,
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    split: str,
) -> EpochResult:
    with torch.inference_mode():
        result, _, _ = run_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            gradient_clip_norm=0.0,
            split=split,
        )
        return result


def save_checkpoint(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau | None,
    epoch: int,
    best_val_f1: float,
    best_epoch: int,
    global_step: int,
    early_stopping_bad_epochs: int,
    history_rows: list[dict[str, Any]],
    train_step_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "global_step": global_step,
        "early_stopping_bad_epochs": early_stopping_bad_epochs,
        "history_rows": history_rows,
        "train_step_rows": train_step_rows,
        "label_names": list(EMOTION_LABELS),
        "config": vars(args),
        "frozen_emotion2vec": True,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
        "cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(checkpoint, checkpoint_path)


def save_predictions(
    result: EpochResult,
    output_path: Path,
) -> None:
    rows = []
    for sample_id, true_id, pred_id, embedding_path in zip(
        result.sample_ids,
        result.y_true,
        result.y_pred,
        result.embedding_paths,
        strict=False,
    ):
        rows.append(
            {
                "sample_id": sample_id,
                "true_label": ID_TO_LABEL[int(true_id)],
                "pred_label": ID_TO_LABEL[int(pred_id)],
                "embedding_path": embedding_path,
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def plot_learning_curves(history_frame: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].plot(history_frame["epoch"], history_frame["train_loss"], label="train")
    axes[0].plot(history_frame["epoch"], history_frame["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy")
    axes[0].legend()

    axes[1].plot(history_frame["epoch"], history_frame["train_accuracy"], label="train")
    axes[1].plot(history_frame["epoch"], history_frame["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    axes[2].plot(history_frame["epoch"], history_frame["train_f1_macro"], label="train")
    axes[2].plot(history_frame["epoch"], history_frame["val_f1_macro"], label="val")
    axes[2].set_title("Macro F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def maybe_init_wandb(
    args: argparse.Namespace,
    run_dir: Path,
    resolved_config: dict[str, Any],
) -> MultiWandbRun | None:
    if args.report_to != "wandb":
        return None

    try:
        import wandb  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "wandb reporting was requested but wandb is not installed in the active environment."
        ) from exc

    return init_multi_wandb_run(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=run_dir.name,
        dir=str(run_dir),
        config=resolved_config,
        tags=[tag.strip() for tag in str(args.wandb_tags).split(",") if tag.strip()],
        mode=args.wandb_mode,
        reinit=True,
    )


def log_epoch_to_console(epoch: int, train_result: EpochResult, val_result: EpochResult) -> None:
    print(
        f"Epoch {epoch:03d} | "
        f"train_loss={train_result.loss:.4f} train_acc={train_result.metrics['accuracy']:.4f} "
        f"train_f1={train_result.metrics['f1_macro']:.4f} | "
        f"val_loss={val_result.loss:.4f} val_acc={val_result.metrics['accuracy']:.4f} "
        f"val_f1={val_result.metrics['f1_macro']:.4f}"
    )


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    ensure_required_args(args)
    if args.early_stopping_patience is not None and args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be >= 1 when provided.")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("--early-stopping-min-delta must be >= 0.0.")
    if args.lr_scheduler_patience < 0:
        raise ValueError("--lr-scheduler-patience must be >= 0.")
    if not 0.0 < args.lr_scheduler_factor < 1.0:
        raise ValueError("--lr-scheduler-factor must be between 0 and 1.")
    if args.lr_scheduler_min_lr < 0.0:
        raise ValueError("--lr-scheduler-min-lr must be >= 0.0.")

    repo_root = find_repo_root(Path(__file__).resolve())
    device = resolve_device(args.device)
    set_random_seed(args.seed)

    run_dir = make_run_dir(args, repo_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = {
        **vars(args),
        "repo_root": str(repo_root),
        "resolved_device": str(device),
        "label_names": list(EMOTION_LABELS),
        "frozen_emotion2vec": True,
        "host": socket.gethostname(),
    }
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_builtin(resolved_config), handle, sort_keys=True)

    train_loader, val_loader, test_loader = create_dataloaders(args=args, repo_root=repo_root, device=device)
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset

    model = Emotion2VecDownstreamClassifier(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=len(EMOTION_LABELS),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model=model, args=args)
    scheduler = build_scheduler(optimizer=optimizer, args=args)

    wandb_run = maybe_init_wandb(args=args, run_dir=run_dir, resolved_config=resolved_config)
    if wandb_run is not None and args.wandb_watch_model:
        try:
            import wandb

            for run in wandb_run.runs:
                run.watch(model, log="gradients", log_freq=max(args.log_every_steps, 10))
        except Exception as exc:  # pragma: no cover - watch should not fail the training run
            print(f"[wandb] model watch skipped: {type(exc).__name__}: {exc}")

    history_rows: list[dict[str, Any]] = []
    train_step_rows: list[dict[str, Any]] = []
    best_val_f1 = float("-inf")
    best_epoch = -1
    global_step = 0
    early_stopping_bad_epochs = 0
    start_epoch = 1
    best_checkpoint_path = run_dir / "best_model.pt"
    last_checkpoint_path = run_dir / "last_model.pt"
    resume_checkpoint_path = resolve_resume_checkpoint(
        requested=args.resume_from_checkpoint,
        run_dir=run_dir,
        repo_root=repo_root,
    )
    if resume_checkpoint_path is not None:
        checkpoint = load_training_checkpoint(resume_checkpoint_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
        scheduler_state_dict = checkpoint.get("scheduler_state_dict")
        if scheduler is not None and scheduler_state_dict is not None:
            scheduler.load_state_dict(scheduler_state_dict)
        history_rows = list(
            checkpoint.get("history_rows") or load_rows_from_csv(run_dir / "history.csv")
        )
        train_step_rows = list(
            checkpoint.get("train_step_rows") or load_rows_from_csv(run_dir / "train_steps.csv")
        )
        best_val_f1 = float(checkpoint.get("best_val_f1", best_val_f1))
        best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        global_step = int(
            checkpoint.get(
                "global_step",
                train_step_rows[-1]["global_step"] if train_step_rows else 0,
            )
        )
        early_stopping_bad_epochs = int(checkpoint.get("early_stopping_bad_epochs", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        restore_random_state(checkpoint)
        print(
            f"[resume] loaded checkpoint={resume_checkpoint_path} "
            f"epoch={checkpoint.get('epoch', 0)} next_epoch={start_epoch} "
            f"best_epoch={best_epoch} best_val_f1={best_val_f1:.4f} global_step={global_step}"
        )

    label_frames = {
        "train": count_labels(train_dataset),
        "val": count_labels(val_dataset),
        "test": count_labels(test_dataset),
    }
    lengths_by_split = {
        "train": collect_frame_lengths(train_dataset),
        "val": collect_frame_lengths(val_dataset),
        "test": collect_frame_lengths(test_dataset),
    }
    label_counts_path = run_dir / "label_counts_by_split.csv"
    length_stats_path = run_dir / "frame_lengths_by_split.csv"
    label_counts_df = pd.concat(
        [frame.assign(split=split_name) for split_name, frame in label_frames.items()],
        ignore_index=True,
    )
    label_counts_df.to_csv(label_counts_path, index=False)
    length_stats_df = pd.concat(
        [
            pd.DataFrame({"split": split_name, "num_frames": lengths})
            for split_name, lengths in lengths_by_split.items()
        ],
        ignore_index=True,
    )
    length_stats_df.to_csv(length_stats_path, index=False)

    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
        train_result, global_step, epoch_step_rows = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=args.gradient_clip_norm,
            split=str(epoch),
            wandb_run=wandb_run,
            log_every_steps=args.log_every_steps,
            starting_global_step=global_step,
            epoch_index=epoch,
        )
        train_step_rows.extend(epoch_step_rows)
        val_result = evaluate_split(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            split="val",
        )

        history_row = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "train_accuracy": train_result.metrics["accuracy"],
            "train_f1_macro": train_result.metrics["f1_macro"],
            "train_f1_weighted": train_result.metrics["f1_weighted"],
            "val_loss": val_result.loss,
            "val_accuracy": val_result.metrics["accuracy"],
            "val_f1_macro": val_result.metrics["f1_macro"],
            "val_f1_weighted": val_result.metrics["f1_weighted"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history_rows.append(history_row)
        write_progress_tables(run_dir=run_dir, history_rows=history_rows, train_step_rows=train_step_rows)
        log_epoch_to_console(epoch=epoch, train_result=train_result, val_result=val_result)

        if wandb_run is not None:
            wandb_run.log(
                {
                    **history_row,
                    "train/epoch_loss": train_result.loss,
                    "train/epoch_accuracy": train_result.metrics["accuracy"],
                    "train/epoch_f1_macro": train_result.metrics["f1_macro"],
                    "val/epoch_loss": val_result.loss,
                    "val/epoch_accuracy": val_result.metrics["accuracy"],
                    "val/epoch_f1_macro": val_result.metrics["f1_macro"],
                    "train/epoch_learning_rate": history_row["learning_rate"],
                },
                step=global_step,
            )

        if scheduler is not None:
            scheduler.step(val_result.metrics["f1_macro"])

        improved = val_result.metrics["f1_macro"] > (best_val_f1 + args.early_stopping_min_delta)
        if improved:
            best_val_f1 = val_result.metrics["f1_macro"]
            best_epoch = epoch
            early_stopping_bad_epochs = 0
            save_checkpoint(
                best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_f1=best_val_f1,
                best_epoch=best_epoch,
                global_step=global_step,
                early_stopping_bad_epochs=early_stopping_bad_epochs,
                history_rows=history_rows,
                train_step_rows=train_step_rows,
                args=args,
            )
        else:
            early_stopping_bad_epochs += 1

        save_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_f1=best_val_f1,
            best_epoch=best_epoch,
            global_step=global_step,
            early_stopping_bad_epochs=early_stopping_bad_epochs,
            history_rows=history_rows,
            train_step_rows=train_step_rows,
            args=args,
        )

        if (
            args.early_stopping_patience is not None
            and early_stopping_bad_epochs >= args.early_stopping_patience
        ):
            stopped_early = True
            print(
                f"[early-stop] stopped at epoch={epoch} "
                f"best_epoch={best_epoch} best_val_f1={best_val_f1:.4f} "
                f"bad_epochs={early_stopping_bad_epochs}"
            )
            break

    history_frame = pd.DataFrame(history_rows)
    train_steps_frame = pd.DataFrame(train_step_rows)
    write_progress_tables(run_dir=run_dir, history_rows=history_rows, train_step_rows=train_step_rows)

    best_checkpoint = load_training_checkpoint(best_checkpoint_path, device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_result = evaluate_split(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        split="test",
    )
    val_result_best = evaluate_split(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        split="val_best",
    )

    test_metrics_path = run_dir / "test_metrics.json"
    test_metrics_payload = {
        "best_epoch": best_epoch,
        "best_val_f1_macro": best_val_f1,
        "stopped_early": stopped_early,
        "val_accuracy": val_result_best.metrics["accuracy"],
        "val_f1_macro": val_result_best.metrics["f1_macro"],
        "val_f1_weighted": val_result_best.metrics["f1_weighted"],
        "test_accuracy": test_result.metrics["accuracy"],
        "test_f1_macro": test_result.metrics["f1_macro"],
        "test_f1_weighted": test_result.metrics["f1_weighted"],
        "test_precision_macro": test_result.metrics["precision_macro"],
        "test_recall_macro": test_result.metrics["recall_macro"],
        "frozen_emotion2vec": True,
    }
    with test_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(to_builtin(test_metrics_payload), handle, indent=2)

    report_frame = build_classification_report_frame(
        y_true=test_result.y_true,
        y_pred=test_result.y_pred,
        label_names=EMOTION_LABELS,
    )
    report_frame.to_csv(run_dir / "classification_report_test.csv", index=False)

    confusion = compute_confusion_matrix(
        y_true=test_result.y_true,
        y_pred=test_result.y_pred,
        num_classes=len(EMOTION_LABELS),
    )
    confusion_normalized = compute_normalized_confusion_matrix(
        y_true=test_result.y_true,
        y_pred=test_result.y_pred,
        num_classes=len(EMOTION_LABELS),
    )
    confusion_frame = pd.DataFrame(confusion, index=EMOTION_LABELS, columns=EMOTION_LABELS)
    confusion_frame.to_csv(run_dir / "confusion_matrix_test.csv")
    confusion_normalized_frame = pd.DataFrame(
        confusion_normalized,
        index=EMOTION_LABELS,
        columns=EMOTION_LABELS,
    )
    confusion_normalized_frame.to_csv(run_dir / "confusion_matrix_test_normalized.csv")
    confusion_long_frame = melt_confusion_matrix(confusion, EMOTION_LABELS, normalized=False)
    confusion_long_frame.to_csv(run_dir / "confusion_matrix_test_long.csv", index=False)
    confusion_norm_long_frame = melt_confusion_matrix(confusion_normalized, EMOTION_LABELS, normalized=True)
    confusion_norm_long_frame.to_csv(run_dir / "confusion_matrix_test_normalized_long.csv", index=False)

    learning_curve_path = run_dir / "learning_curves.png"
    confusion_path = run_dir / "confusion_matrix_test.png"
    confusion_normalized_path = run_dir / "confusion_matrix_test_normalized.png"
    label_distribution_path = run_dir / "label_distribution.png"
    frame_lengths_path = run_dir / "frame_length_histogram.png"
    per_class_f1_path = run_dir / "per_class_f1.png"
    per_class_metrics_path = run_dir / "per_class_metrics.png"
    prediction_distribution_path = run_dir / "test_true_vs_pred_distribution.png"

    if args.save_figures:
        plot_learning_curves(history_frame=history_frame, output_path=learning_curve_path)
        plot_confusion_matrix(
            matrix=confusion,
            label_names=EMOTION_LABELS,
            output_path=confusion_path,
            title="Emotion2Vec Downstream SER Confusion Matrix",
        )
        plot_confusion_matrix(
            matrix=confusion_normalized,
            label_names=EMOTION_LABELS,
            output_path=confusion_normalized_path,
            title="Emotion2Vec Downstream SER Confusion Matrix (Normalized)",
            fmt=".2f",
            value_label="True Label",
        )
        plot_label_distribution(label_frames=label_frames, output_path=label_distribution_path)
        plot_length_histogram(lengths_by_split=lengths_by_split, output_path=frame_lengths_path)
        plot_per_class_f1(report_frame=report_frame, output_path=per_class_f1_path)
        plot_per_class_metrics(report_frame=report_frame, output_path=per_class_metrics_path)
        plot_true_vs_pred_distribution(result=test_result, output_path=prediction_distribution_path)

    if args.save_predictions:
        save_predictions(result=test_result, output_path=run_dir / "test_predictions.csv")

    if wandb_run is not None:
        wandb_run.summary["best_epoch"] = int(best_epoch)
        wandb_run.summary["best_val_f1_macro"] = float(best_val_f1)
        for key, value in test_metrics_payload.items():
            wandb_run.summary[key] = value

        try:
            import wandb

            wandb_run.log({"history_table": wandb.Table(dataframe=history_frame)})
            wandb_run.log({"train_steps": wandb.Table(dataframe=train_steps_frame)})
            wandb_run.log({"classification_report_test": wandb.Table(dataframe=report_frame)})
            wandb_run.log({"label_counts_by_split": wandb.Table(dataframe=label_counts_df)})
            wandb_run.log({"frame_lengths_by_split": wandb.Table(dataframe=length_stats_df)})
            wandb_run.log({"confusion_matrix_test": wandb.Table(dataframe=confusion_long_frame)})
            wandb_run.log({"confusion_matrix_test_normalized": wandb.Table(dataframe=confusion_norm_long_frame)})
            if args.save_predictions:
                predictions_df = pd.read_csv(run_dir / "test_predictions.csv")
                wandb_run.log({"test_predictions": wandb.Table(dataframe=predictions_df)})
            if args.save_figures:
                wandb_run.log(
                    {
                        "learning_curves": wandb.Image(str(learning_curve_path)),
                        "confusion_matrix_test": wandb.Image(str(confusion_path)),
                        "confusion_matrix_test_normalized": wandb.Image(str(confusion_normalized_path)),
                        "label_distribution": wandb.Image(str(label_distribution_path)),
                        "frame_length_histogram": wandb.Image(str(frame_lengths_path)),
                        "per_class_f1": wandb.Image(str(per_class_f1_path)),
                        "per_class_metrics": wandb.Image(str(per_class_metrics_path)),
                        "test_true_vs_pred_distribution": wandb.Image(str(prediction_distribution_path)),
                    }
                )
        except Exception as exc:  # pragma: no cover - logging should not fail the training run
            print(f"[wandb] artifact logging skipped: {type(exc).__name__}: {exc}")
        finally:
            wandb_run.finish()

    print(
        f"Best epoch: {best_epoch} | "
        f"best_val_f1={best_val_f1:.4f} | "
        f"test_acc={test_result.metrics['accuracy']:.4f} | "
        f"test_f1={test_result.metrics['f1_macro']:.4f}"
    )
    print(f"Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
