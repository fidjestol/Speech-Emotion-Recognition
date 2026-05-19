from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def compute_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
) -> np.ndarray:
    if SKLEARN_AVAILABLE:
        return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred, strict=False):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def compute_normalized_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
) -> np.ndarray:
    matrix = compute_confusion_matrix(y_true=y_true, y_pred=y_pred, num_classes=num_classes).astype(np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Sequence[str],
) -> dict[str, float]:
    label_ids = list(range(len(label_names)))

    if SKLEARN_AVAILABLE:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(
                precision_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)
            ),
            "f1_macro": float(
                f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)
            ),
            "precision_weighted": float(
                precision_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)
            ),
            "recall_weighted": float(
                recall_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)
            ),
        }

    matrix = compute_confusion_matrix(y_true=y_true, y_pred=y_pred, num_classes=len(label_names))
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)

    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=(precision + recall) != 0,
    )

    total = max(int(support.sum()), 1)
    weights = support / total

    accuracy = float(true_positive.sum() / total)
    return {
        "accuracy": accuracy,
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "precision_weighted": float(np.sum(weights * precision)),
        "recall_weighted": float(np.sum(weights * recall)),
        "f1_weighted": float(np.sum(weights * f1)),
    }


def build_classification_report_frame(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Sequence[str],
) -> pd.DataFrame:
    label_ids = list(range(len(label_names)))

    if SKLEARN_AVAILABLE:
        report = classification_report(
            y_true,
            y_pred,
            labels=label_ids,
            target_names=list(label_names),
            zero_division=0,
            output_dict=True,
        )
        return pd.DataFrame(report).transpose().reset_index(names="label")

    metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred, label_names=label_names)
    matrix = compute_confusion_matrix(y_true=y_true, y_pred=y_pred, num_classes=len(label_names))
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)

    rows: list[dict[str, Any]] = []
    for idx, label_name in enumerate(label_names):
        precision = float(true_positive[idx] / predicted[idx]) if predicted[idx] else 0.0
        recall = float(true_positive[idx] / support[idx]) if support[idx] else 0.0
        f1 = 0.0 if precision + recall == 0.0 else float((2 * precision * recall) / (precision + recall))
        rows.append(
            {
                "label": label_name,
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "support": int(support[idx]),
            }
        )

    rows.append(
        {
            "label": "accuracy",
            "precision": metrics["accuracy"],
            "recall": metrics["accuracy"],
            "f1-score": metrics["accuracy"],
            "support": int(support.sum()),
        }
    )
    rows.append(
        {
            "label": "macro avg",
            "precision": metrics["precision_macro"],
            "recall": metrics["recall_macro"],
            "f1-score": metrics["f1_macro"],
            "support": int(support.sum()),
        }
    )
    rows.append(
        {
            "label": "weighted avg",
            "precision": metrics["precision_weighted"],
            "recall": metrics["recall_weighted"],
            "f1-score": metrics["f1_weighted"],
            "support": int(support.sum()),
        }
    )
    return pd.DataFrame(rows)


def plot_confusion_matrix(
    matrix: np.ndarray,
    label_names: Sequence[str],
    output_path: str | Path,
    *,
    title: str,
    fmt: str = "d",
    value_label: str = "True",
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=list(label_names),
        yticklabels=list(label_names),
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel(value_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
