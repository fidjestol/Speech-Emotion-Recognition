from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from feature_selection.common import DEFAULT_USE_VARIANT_ARTIFACT_DIRS, variant_name

DEFAULT_AUGMENTATION_METHOD = "class_conditional_mean_std_oversampling"
DEFAULT_GROUP_SIZE = 5
DEFAULT_RANDOM_STATE = 42
DEFAULT_TARGET_PER_CLASS = 10_000
DEFAULT_SAMPLE_WITH_REPLACEMENT = True


def resolve_augmentation_artifact_dir(
    repo_root: Path,
    *,
    include_xxx: bool,
    use_variant_dirs: bool = DEFAULT_USE_VARIANT_ARTIFACT_DIRS,
    run_name: str | None = None,
    dataset_key: str | None = None,
) -> Path:
    artifact_dir = repo_root / "feature_augmentation" / "mean_std_oversampling" / "artifacts"
    if run_name:
        artifact_dir = artifact_dir / str(run_name)
    if use_variant_dirs:
        artifact_dir = artifact_dir / variant_name(include_xxx)
    if dataset_key:
        artifact_dir = artifact_dir / str(dataset_key)
    return artifact_dir


def resolve_prototype_artifact_dir(
    repo_root: Path,
    *,
    include_xxx: bool,
    use_variant_dirs: bool = DEFAULT_USE_VARIANT_ARTIFACT_DIRS,
) -> Path:
    return resolve_augmentation_artifact_dir(
        repo_root,
        include_xxx=include_xxx,
        use_variant_dirs=use_variant_dirs,
    )


def summarize_class_counts(labels: Iterable[object], *, label_name: str = "emotion") -> pd.DataFrame:
    series = pd.Series(list(labels), name=label_name).astype(str)
    counts = (
        series.value_counts(dropna=False)
        .rename_axis(label_name)
        .reset_index(name="count")
        .sort_values(label_name, ignore_index=True)
    )
    return counts


@dataclass(frozen=True)
class MeanStdOversamplingConfig:
    target_per_class: int = DEFAULT_TARGET_PER_CLASS
    group_size: int = DEFAULT_GROUP_SIZE
    random_state: int = DEFAULT_RANDOM_STATE
    sample_with_replacement: bool = DEFAULT_SAMPLE_WITH_REPLACEMENT


PrototypeOversamplingConfig = MeanStdOversamplingConfig


def _validate_training_inputs(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must contain the same number of rows.")
    if X_train.empty:
        raise ValueError("X_train is empty.")
    if y_train.empty:
        raise ValueError("y_train is empty.")

    work_X = X_train.reset_index(drop=True).copy()
    work_y = pd.Series(y_train).reset_index(drop=True).astype(str)

    if work_X.isna().any().any():
        raise ValueError("X_train contains NaN values. Clean the frame before augmentation.")
    return work_X, work_y


def mean_std_oversample_training_frame(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    target_per_class: int = DEFAULT_TARGET_PER_CLASS,
    group_size: int = DEFAULT_GROUP_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    sample_with_replacement: bool = DEFAULT_SAMPLE_WITH_REPLACEMENT,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    work_X, work_y = _validate_training_inputs(X_train, y_train)
    feature_names = list(work_X.columns)
    rng = np.random.default_rng(random_state)

    synthetic_rows: list[np.ndarray] = []
    synthetic_labels: list[str] = []
    synthetic_meta_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    prototype_group_id = 0

    for class_label in sorted(work_y.unique()):
        class_mask = work_y == class_label
        class_values = work_X.loc[class_mask, feature_names].to_numpy(dtype=float, copy=True)
        original_count = int(class_values.shape[0])
        if original_count == 0:
            continue

        target_count = max(int(target_per_class), original_count)
        synthetic_needed = target_count - original_count
        synthetic_added = 0

        while synthetic_added < synthetic_needed:
            replace = bool(sample_with_replacement or original_count < group_size)
            chosen_indices = rng.choice(original_count, size=int(group_size), replace=replace)
            chosen = class_values[chosen_indices]

            prototype_mean = chosen.mean(axis=0)
            prototype_std = chosen.std(axis=0, ddof=0)

            candidates = (
                ("mean_minus_std", prototype_mean - prototype_std),
                ("mean", prototype_mean),
                ("mean_plus_std", prototype_mean + prototype_std),
            )

            for prototype_kind, values in candidates:
                if synthetic_added >= synthetic_needed:
                    break
                synthetic_rows.append(values.astype(float, copy=False))
                synthetic_labels.append(class_label)
                synthetic_meta_rows.append(
                    {
                        "emotion": class_label,
                        "synthetic_group_id": int(prototype_group_id),
                        "synthetic_kind": str(prototype_kind),
                        "prototype_group_id": int(prototype_group_id),
                        "prototype_kind": str(prototype_kind),
                        "group_size": int(group_size),
                        "synthetic_source": DEFAULT_AUGMENTATION_METHOD,
                    }
                )
                synthetic_added += 1

            prototype_group_id += 1

        summary_rows.append(
            {
                "emotion": class_label,
                "original_count": int(original_count),
                "synthetic_added": int(synthetic_added),
                "target_count": int(target_count),
                "final_count": int(original_count + synthetic_added),
            }
        )

    synthetic_X = pd.DataFrame(synthetic_rows, columns=feature_names)
    synthetic_y = pd.Series(synthetic_labels, name=y_train.name or "emotion")
    synthetic_meta = pd.DataFrame(
        synthetic_meta_rows,
        columns=[
            "emotion",
            "synthetic_group_id",
            "synthetic_kind",
            "prototype_group_id",
            "prototype_kind",
            "group_size",
            "synthetic_source",
        ],
    )
    summary_df = pd.DataFrame(
        summary_rows,
        columns=["emotion", "original_count", "synthetic_added", "target_count", "final_count"],
    ).sort_values("emotion", ignore_index=True)

    X_aug = pd.concat([work_X, synthetic_X], axis=0, ignore_index=True)
    y_aug = pd.concat(
        [work_y.rename(y_train.name or "emotion"), synthetic_y],
        axis=0,
        ignore_index=True,
    )
    return X_aug, y_aug, synthetic_meta, summary_df


def prototype_oversample_training_frame(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    target_per_class: int = DEFAULT_TARGET_PER_CLASS,
    group_size: int = DEFAULT_GROUP_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    sample_with_replacement: bool = DEFAULT_SAMPLE_WITH_REPLACEMENT,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    return mean_std_oversample_training_frame(
        X_train,
        y_train,
        target_per_class=target_per_class,
        group_size=group_size,
        random_state=random_state,
        sample_with_replacement=sample_with_replacement,
    )


def build_stage_class_count_table(
    stage_to_labels: dict[str, Iterable[object]],
    *,
    label_name: str = "emotion",
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for stage_name, labels in stage_to_labels.items():
        frame = summarize_class_counts(labels, label_name=label_name).rename(columns={"count": "class_count"})
        frame.insert(0, "stage", str(stage_name))
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["stage", label_name, "class_count"])
    return pd.concat(frames, ignore_index=True)


def build_classification_report_frame(
    report_dict: dict[str, object],
    *,
    stage_name: str,
    model_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label_name, values in report_dict.items():
        if not isinstance(values, dict):
            continue
        row = {
            "stage": str(stage_name),
            "model": str(model_name),
            "label": str(label_name),
            "precision": float(values.get("precision", 0.0)),
            "recall": float(values.get("recall", 0.0)),
            "f1_score": float(values.get("f1-score", 0.0)),
            "support": int(values.get("support", 0)),
        }
        row["label_type"] = "class" if label_name not in {"macro avg", "weighted avg", "micro avg"} else "summary"
        rows.append(row)
    return pd.DataFrame(rows)


def build_confusion_matrix_frame(
    confusion_matrix_values: np.ndarray,
    *,
    labels: Iterable[object],
    stage_name: str,
    model_name: str,
) -> pd.DataFrame:
    ordered_labels = [str(label) for label in labels]
    rows: list[dict[str, object]] = []
    for true_idx, true_label in enumerate(ordered_labels):
        for pred_idx, pred_label in enumerate(ordered_labels):
            rows.append(
                {
                    "stage": str(stage_name),
                    "model": str(model_name),
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "count": int(confusion_matrix_values[true_idx, pred_idx]),
                }
            )
    return pd.DataFrame(rows)


def build_normalized_confusion_matrix_frame(
    confusion_matrix_values: np.ndarray,
    *,
    labels: Iterable[object],
    stage_name: str,
    model_name: str,
) -> pd.DataFrame:
    ordered_labels = [str(label) for label in labels]
    rows: list[dict[str, object]] = []
    for true_idx, true_label in enumerate(ordered_labels):
        row_sum = float(confusion_matrix_values[true_idx].sum())
        for pred_idx, pred_label in enumerate(ordered_labels):
            value = 0.0 if row_sum == 0 else float(confusion_matrix_values[true_idx, pred_idx]) / row_sum
            rows.append(
                {
                    "stage": str(stage_name),
                    "model": str(model_name),
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "normalized_value": value,
                }
            )
    return pd.DataFrame(rows)


def compute_binary_classification_stats(
    confusion_matrix_values: np.ndarray,
    *,
    labels: Iterable[object],
    stage_name: str,
    model_name: str,
) -> pd.DataFrame:
    ordered_labels = [str(label) for label in labels]
    total = int(confusion_matrix_values.sum())
    rows: list[dict[str, object]] = []
    for idx, label in enumerate(ordered_labels):
        tp = int(confusion_matrix_values[idx, idx])
        fn = int(confusion_matrix_values[idx, :].sum() - tp)
        fp = int(confusion_matrix_values[:, idx].sum() - tp)
        tn = int(total - tp - fp - fn)
        one_vs_rest_accuracy = 0.0 if total == 0 else float(tp + tn) / float(total)
        class_accuracy = 0.0 if tp + fn == 0 else float(tp) / float(tp + fn)
        rows.append(
            {
                "stage": str(stage_name),
                "model": str(model_name),
                "label": label,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "class_accuracy": class_accuracy,
                "one_vs_rest_accuracy": one_vs_rest_accuracy,
            }
        )
    return pd.DataFrame(rows)
