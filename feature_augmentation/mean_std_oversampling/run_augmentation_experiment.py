from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_augmentation.common import (  # noqa: E402
    DEFAULT_AUGMENTATION_METHOD,
    build_classification_report_frame,
    build_confusion_matrix_frame,
    build_normalized_confusion_matrix_frame,
    build_stage_class_count_table,
    compute_binary_classification_stats,
    mean_std_oversample_training_frame,
    resolve_augmentation_artifact_dir,
    summarize_class_counts,
)
from feature_selection.common import (  # noqa: E402
    DEFAULT_EXCLUDED_EMOTIONS,
    DEFAULT_REQUIRE_AGREEMENT,
    METADATA_CANDIDATES,
    filter_feature_frame,
    machine_name_from_env,
    resolve_cpu_parallel_config,
    resolve_feature_source,
    variant_name,
)
from utils.wandb_multi import init_multi_wandb_run

plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

ALL_MODEL_NAMES = [
    "logreg",
    "linear_svc_cal",
    "polynomial_svc_cal",
    "xgboost",
    "catboost",
    "soft_voting",
    "stacking",
]


def is_metadata_column(column_name: str) -> bool:
    if column_name in METADATA_CANDIDATES:
        return True
    if "__" in column_name:
        suffix = column_name.rsplit("__", 1)[-1]
        return suffix in METADATA_CANDIDATES
    return False


def find_column(columns: list[str], target_name: str) -> str | None:
    if target_name in columns:
        return target_name
    suffix_matches = [column for column in columns if column.rsplit("__", 1)[-1] == target_name]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return None


@dataclass
class ExperimentArgs:
    dataset_key: str
    run_name: str
    include_xxx: bool
    use_variant_dirs: bool
    require_agreement: bool
    excluded_emotions: list[str]
    target_col: str
    session_col: str
    test_size: float
    random_state: int
    target_per_class: int
    group_size: int
    augmentation_random_state: int
    use_nrows: int | None
    enable_gpu_models: bool
    report_to_wandb: bool
    wandb_project: str
    wandb_entity: str | None
    wandb_group: str | None
    save_augmented_matrices: bool
    save_models: bool
    models: list[str]
    xgb_estimators: int
    xgb_learning_rate: float
    xgb_max_depth: int
    xgb_subsample: float
    xgb_colsample_bytree: float
    poly_degree: int
    poly_coef0: float
    poly_c: float


class EncodedXGBClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(
        self,
        *,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.9,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = 1,
        use_gpu: bool = False,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.model_: Any | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EncodedXGBClassifier":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # noqa: F401
            raise RuntimeError("xgboost is required for EncodedXGBClassifier. Install project dependencies first.") from exc

        y_array = np.asarray(y).astype(str)
        self.classes_, encoded = np.unique(y_array, return_inverse=True)
        params: dict[str, Any] = {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "num_class": len(self.classes_),
            "tree_method": "hist",
        }
        if self.use_gpu:
            params["device"] = "cuda"
        self.model_ = XGBClassifier(**params)
        self.model_.fit(np.asarray(X, dtype=np.float32), encoded)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None or self.classes_ is None:
            raise RuntimeError("EncodedXGBClassifier has not been fitted.")
        encoded = self.model_.predict(np.asarray(X, dtype=np.float32)).astype(int)
        return self.classes_[encoded]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None or self.classes_ is None:
            raise RuntimeError("EncodedXGBClassifier has not been fitted.")
        return np.asarray(self.model_.predict_proba(np.asarray(X, dtype=np.float32)), dtype=float)


class EncodedCatBoostClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(
        self,
        *,
        iterations: int = 400,
        learning_rate: float = 0.05,
        depth: int = 6,
        random_state: int = 42,
        thread_count: int = 1,
        use_gpu: bool = False,
    ) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_state = random_state
        self.thread_count = thread_count
        self.use_gpu = use_gpu
        self.model_: Any | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EncodedCatBoostClassifier":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise RuntimeError("catboost is required for EncodedCatBoostClassifier. Install project dependencies first.") from exc

        y_array = np.asarray(y).astype(str)
        self.classes_, encoded = np.unique(y_array, return_inverse=True)
        params: dict[str, Any] = {
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "random_seed": self.random_state,
            "thread_count": self.thread_count,
            "verbose": False,
            "allow_writing_files": False,
        }
        if self.use_gpu:
            params["task_type"] = "GPU"
            params["devices"] = "0"
        self.model_ = CatBoostClassifier(**params)
        self.model_.fit(np.asarray(X, dtype=np.float32), encoded)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None or self.classes_ is None:
            raise RuntimeError("EncodedCatBoostClassifier has not been fitted.")
        encoded = np.asarray(self.model_.predict(np.asarray(X, dtype=np.float32))).reshape(-1).astype(int)
        return self.classes_[encoded]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None or self.classes_ is None:
            raise RuntimeError("EncodedCatBoostClassifier has not been fitted.")
        return np.asarray(self.model_.predict_proba(np.asarray(X, dtype=np.float32)), dtype=float)


class CalibratedPipelineClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(self, estimator: Any, *, cv: int = 3, method: str = "sigmoid") -> None:
        self.estimator = estimator
        self.cv = cv
        self.method = method
        self.model_: CalibratedClassifierCV | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CalibratedPipelineClassifier":
        self.model_ = CalibratedClassifierCV(estimator=clone(self.estimator), cv=self.cv, method=self.method)
        self.model_.fit(X, y.astype(str))
        self.classes_ = np.asarray(self.model_.classes_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("CalibratedPipelineClassifier has not been fitted.")
        return self.model_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("CalibratedPipelineClassifier has not been fitted.")
        return self.model_.predict_proba(X)


ModelBuilder = Callable[[], Any]
ProgressHook = Callable[[dict[str, Any]], None]


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def environment_snapshot(*, probe_xgboost_gpu: bool = False) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python": sys.version,
        "git_commit": current_git_commit(),
        "xgboost_gpu_requested": probe_xgboost_gpu,
    }
    return snapshot


def safe_predict_proba(model: Any, X_eval: pd.DataFrame, classes: list[str]) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X_eval), dtype=float)
        if proba.ndim == 2 and proba.shape[1] == len(classes):
            return proba
    return None


def build_model_builders(
    *,
    random_state: int,
    model_parallel_jobs: int,
    enable_gpu_models: bool,
    xgb_estimators: int,
    xgb_learning_rate: float,
    xgb_max_depth: int,
    xgb_subsample: float,
    xgb_colsample_bytree: float,
    poly_degree: int,
    poly_coef0: float,
    poly_c: float,
) -> dict[str, ModelBuilder]:
    probabilistic_builders: dict[str, ModelBuilder] = {
        "logreg": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "linear_svc_cal": lambda: CalibratedPipelineClassifier(
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LinearSVC(class_weight="balanced", random_state=random_state, max_iter=8000)),
                ]
            ),
            cv=3,
            method="sigmoid",
        ),
        "polynomial_svc_cal": lambda: CalibratedPipelineClassifier(
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        SVC(
                            kernel="poly",
                            degree=poly_degree,
                            coef0=poly_coef0,
                            C=poly_c,
                            gamma="scale",
                            class_weight="balanced",
                            random_state=random_state,
                            cache_size=1024,
                        ),
                    ),
                ]
            ),
            cv=3,
            method="sigmoid",
        ),
        "xgboost": lambda: EncodedXGBClassifier(
            n_estimators=xgb_estimators,
            learning_rate=xgb_learning_rate,
            max_depth=xgb_max_depth,
            subsample=xgb_subsample,
            colsample_bytree=xgb_colsample_bytree,
            random_state=random_state,
            n_jobs=model_parallel_jobs if model_parallel_jobs != 0 else 1,
            use_gpu=enable_gpu_models,
        ),
        "catboost": lambda: EncodedCatBoostClassifier(
            iterations=xgb_estimators,
            learning_rate=xgb_learning_rate,
            depth=xgb_max_depth,
            random_state=random_state,
            thread_count=model_parallel_jobs if model_parallel_jobs != 0 else 1,
            use_gpu=enable_gpu_models,
        ),
    }

    def soft_voting_builder() -> VotingClassifier:
        return VotingClassifier(
            estimators=[
                ("logreg", probabilistic_builders["logreg"]()),
                ("linear_svc_cal", probabilistic_builders["linear_svc_cal"]()),
                ("polynomial_svc_cal", probabilistic_builders["polynomial_svc_cal"]()),
                ("xgboost", probabilistic_builders["xgboost"]()),
                ("catboost", probabilistic_builders["catboost"]()),
            ],
            voting="soft",
            weights=[1.0, 1.0, 1.0, 2.0, 2.0],
            n_jobs=None,
        )

    def stacking_builder() -> StackingClassifier:
        return StackingClassifier(
            estimators=[
                ("logreg", probabilistic_builders["logreg"]()),
                ("linear_svc_cal", probabilistic_builders["linear_svc_cal"]()),
                ("polynomial_svc_cal", probabilistic_builders["polynomial_svc_cal"]()),
                ("xgboost", probabilistic_builders["xgboost"]()),
                ("catboost", probabilistic_builders["catboost"]()),
            ],
            final_estimator=LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state),
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
            stack_method="predict_proba",
            passthrough=False,
        )

    return {
        **probabilistic_builders,
        "soft_voting": soft_voting_builder,
        "stacking": stacking_builder,
    }


def make_wandb_run(args: ExperimentArgs, *, dataset_key: str, variant: str, artifact_dir: Path):
    if not args.report_to_wandb:
        return None
    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "group": args.wandb_group or args.run_name,
        "job_type": "feature_augmentation_dataset",
        "name": f"{args.run_name}_{variant}_{dataset_key}",
        "config": asdict(args),
        "reinit": True,
    }
    saved_run_id = load_saved_wandb_run_id(artifact_dir)
    if saved_run_id is not None:
        init_kwargs["id"] = saved_run_id
        init_kwargs["resume"] = "allow"

    run = init_multi_wandb_run(**init_kwargs)
    save_wandb_run_state(artifact_dir, run)
    return run


def save_figure(path: Path, fig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def wandb_state_path(base_dir: Path) -> Path:
    return base_dir / "wandb_run.json"


def load_saved_wandb_run_id(base_dir: Path) -> str | None:
    path = wandb_state_path(base_dir)
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    run_id = str(payload.get("id", "")).strip()
    return run_id or None


def save_wandb_run_state(base_dir: Path, run: Any) -> None:
    write_json(
        wandb_state_path(base_dir),
        {
            "id": str(getattr(run, "id", "")),
            "name": str(getattr(run, "name", "")),
            "url": getattr(run, "url", None),
            "project": getattr(run, "project", None),
            "entity": getattr(run, "entity", None),
        },
    )


def save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Any:
    return joblib.load(path)


def checkpoint_dir_for_fold(reports_dir: Path, fold_label: str) -> Path:
    return reports_dir / "fold_checkpoints" / fold_label


def fold_model_path(models_dir: Path, stage_name: str, fold_label: str, model_name: str) -> Path:
    return models_dir / stage_name / fold_label / f"{model_name}.joblib"


def final_model_path(model_dir: Path, stage_name: str, model_name: str) -> Path:
    return model_dir / stage_name / f"{model_name}.joblib"


def save_fold_checkpoint(
    *,
    checkpoint_dir: Path,
    dataset_key: str,
    fold_label: str,
    held_out_session: int,
    baseline_results: pd.DataFrame,
    augmented_results: pd.DataFrame,
    baseline_reports: list[pd.DataFrame],
    augmented_reports: list[pd.DataFrame],
    baseline_confs: list[pd.DataFrame],
    augmented_confs: list[pd.DataFrame],
    baseline_confs_norm: list[pd.DataFrame],
    augmented_confs_norm: list[pd.DataFrame],
    baseline_binary: list[pd.DataFrame],
    augmented_binary: list[pd.DataFrame],
    baseline_predictions: list[dict[str, Any]],
    augmented_predictions: list[dict[str, Any]],
    baseline_saved: list[dict[str, Any]],
    augmented_saved: list[dict[str, Any]],
    augmentation_summary_df: pd.DataFrame,
    fold_stage_counts: list[pd.DataFrame],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([baseline_results, augmented_results], ignore_index=True).to_csv(
        checkpoint_dir / f"{dataset_key}_fold_metrics.csv",
        index=False,
    )
    pd.concat([*baseline_reports, *augmented_reports], ignore_index=True).to_csv(
        checkpoint_dir / f"{dataset_key}_classification_reports.csv",
        index=False,
    )
    pd.concat([*baseline_confs, *augmented_confs], ignore_index=True).to_csv(
        checkpoint_dir / f"{dataset_key}_confusion_matrices.csv",
        index=False,
    )
    pd.concat([*baseline_confs_norm, *augmented_confs_norm], ignore_index=True).to_csv(
        checkpoint_dir / f"{dataset_key}_confusion_matrices_normalized.csv",
        index=False,
    )
    pd.concat([*baseline_binary, *augmented_binary], ignore_index=True).to_csv(
        checkpoint_dir / f"{dataset_key}_per_class_accuracy.csv",
        index=False,
    )
    pd.DataFrame([*baseline_predictions, *augmented_predictions]).to_csv(
        checkpoint_dir / f"{dataset_key}_fold_predictions.csv",
        index=False,
    )
    pd.DataFrame([*baseline_saved, *augmented_saved]).to_csv(
        checkpoint_dir / f"{dataset_key}_saved_models.csv",
        index=False,
    )
    augmentation_summary_df.to_csv(
        checkpoint_dir / f"{dataset_key}_augmentation_summary_by_fold.csv",
        index=False,
    )
    pd.concat(fold_stage_counts, ignore_index=True).to_csv(
        checkpoint_dir / f"{dataset_key}_fold_class_counts.csv",
        index=False,
    )
    write_json(
        checkpoint_dir / "fold_complete.json",
        {
            "dataset_key": dataset_key,
            "fold": fold_label,
            "held_out_session": held_out_session,
            "status": "completed",
        },
    )


def load_fold_checkpoint(*, checkpoint_dir: Path, dataset_key: str) -> dict[str, pd.DataFrame]:
    return {
        "fold_metrics": pd.read_csv(checkpoint_dir / f"{dataset_key}_fold_metrics.csv"),
        "classification_reports": pd.read_csv(checkpoint_dir / f"{dataset_key}_classification_reports.csv"),
        "confusion_matrices": pd.read_csv(checkpoint_dir / f"{dataset_key}_confusion_matrices.csv"),
        "confusion_matrices_normalized": pd.read_csv(checkpoint_dir / f"{dataset_key}_confusion_matrices_normalized.csv"),
        "per_class_accuracy": pd.read_csv(checkpoint_dir / f"{dataset_key}_per_class_accuracy.csv"),
        "fold_predictions": pd.read_csv(checkpoint_dir / f"{dataset_key}_fold_predictions.csv"),
        "saved_models": pd.read_csv(checkpoint_dir / f"{dataset_key}_saved_models.csv"),
        "augmentation_summary_by_fold": pd.read_csv(checkpoint_dir / f"{dataset_key}_augmentation_summary_by_fold.csv"),
        "fold_class_counts": pd.read_csv(checkpoint_dir / f"{dataset_key}_fold_class_counts.csv"),
        "meta": pd.DataFrame([read_json(checkpoint_dir / "fold_complete.json")]),
    }


def plot_class_distributions(raw_counts: pd.DataFrame, filtered_counts: pd.DataFrame, *, target_col: str, variant: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(raw_counts[target_col], raw_counts["count"])
    axes[0].set_title("Raw Class Distribution")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(filtered_counts[target_col], filtered_counts["count"])
    axes[1].set_title(f"Filtered Class Distribution ({variant})")
    axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig


def plot_fold_metric_summary(summary_df: pd.DataFrame, *, metric_name: str, title: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot = summary_df.pivot(index="model", columns="stage", values=metric_name).sort_index()
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    return fig


def plot_session_metric_heatmap(fold_metrics_df: pd.DataFrame, *, metric_name: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = fold_metrics_df.pivot_table(index="model", columns="held_out_session", values=metric_name, aggfunc="mean")
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues", ax=ax)
    ax.set_title(f"{metric_name} by Held-out Session")
    plt.tight_layout()
    return fig


def plot_confusion_heatmaps(y_true: pd.Series, y_pred: pd.Series, *, labels: list[str], title_prefix: str):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f"{title_prefix} Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title(f"{title_prefix} Normalized")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    plt.tight_layout()
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one feature augmentation experiment.")
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--include-xxx", action="store_true")
    parser.add_argument("--use-variant-dirs", action="store_true")
    parser.add_argument(
        "--require-agreement",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REQUIRE_AGREEMENT,
    )
    parser.add_argument("--excluded-emotions", nargs="*", default=list(DEFAULT_EXCLUDED_EMOTIONS))
    parser.add_argument("--target-col", default="emotion")
    parser.add_argument("--session-col", default="session")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--target-per-class", type=int, default=10_000)
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--augmentation-random-state", type=int, default=42)
    parser.add_argument("--use-nrows", type=int)
    parser.add_argument(
        "--enable-gpu-models",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--wandb-project", default="ser-feature-augmentation")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument(
        "--save-augmented-matrices",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--save-models",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--models", nargs="*", default=list(ALL_MODEL_NAMES), choices=ALL_MODEL_NAMES)
    parser.add_argument("--xgb-estimators", type=int, default=400)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-subsample", type=float, default=0.9)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--poly-degree", type=int, default=3)
    parser.add_argument("--poly-coef0", type=float, default=1.0)
    parser.add_argument("--poly-c", type=float, default=1.0)
    return parser


def parse_args(argv: list[str] | None = None) -> ExperimentArgs:
    args = build_parser().parse_args(argv)
    return ExperimentArgs(
        dataset_key=args.dataset_key,
        run_name=args.run_name,
        include_xxx=bool(args.include_xxx),
        use_variant_dirs=bool(args.use_variant_dirs),
        require_agreement=bool(args.require_agreement),
        excluded_emotions=list(args.excluded_emotions),
        target_col=args.target_col,
        session_col=args.session_col,
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        target_per_class=int(args.target_per_class),
        group_size=int(args.group_size),
        augmentation_random_state=int(args.augmentation_random_state),
        use_nrows=args.use_nrows,
        enable_gpu_models=bool(args.enable_gpu_models),
        report_to_wandb="wandb" in {part.strip() for part in args.report_to.split(",") if part.strip()},
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        save_augmented_matrices=bool(args.save_augmented_matrices),
        save_models=bool(args.save_models),
        models=list(args.models),
        xgb_estimators=int(args.xgb_estimators),
        xgb_learning_rate=float(args.xgb_learning_rate),
        xgb_max_depth=int(args.xgb_max_depth),
        xgb_subsample=float(args.xgb_subsample),
        xgb_colsample_bytree=float(args.xgb_colsample_bytree),
        poly_degree=int(args.poly_degree),
        poly_coef0=float(args.poly_coef0),
        poly_c=float(args.poly_c),
    )


def fit_models_for_fold(
    *,
    model_builders: dict[str, ModelBuilder],
    X_train_eval: pd.DataFrame,
    y_train_eval: pd.Series,
    X_test_eval: pd.DataFrame,
    y_test_eval: pd.Series,
    stage_name: str,
    fold_label: str,
    held_out_session: Any,
    all_labels: list[str],
    models_dir: Path,
    save_models_flag: bool,
    progress_hook: ProgressHook | None = None,
) -> tuple[pd.DataFrame, list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    report_frames: list[pd.DataFrame] = []
    conf_frames: list[pd.DataFrame] = []
    conf_norm_frames: list[pd.DataFrame] = []
    binary_frames: list[pd.DataFrame] = []
    prediction_rows: list[dict[str, Any]] = []
    saved_models: list[dict[str, Any]] = []

    for model_name, builder in model_builders.items():
        model_path = fold_model_path(models_dir, stage_name, fold_label, model_name)
        resumed_from_disk = False
        elapsed = float("nan")

        if save_models_flag and model_path.exists():
            try:
                model = load_model(model_path)
                resumed_from_disk = True
                if progress_hook is not None:
                    progress_hook(
                        {
                            "event": "model_reused",
                            "stage": stage_name,
                            "fold": fold_label,
                            "held_out_session": held_out_session,
                            "model": model_name,
                            "model_path": str(model_path),
                        }
                    )
            except Exception as exc:
                print(
                    f"[resume] dataset_model={model_name} fold={fold_label} stage={stage_name} "
                    f"status=load_failed error={type(exc).__name__}: {exc} -> refit",
                    flush=True,
                )

        if not resumed_from_disk:
            if progress_hook is not None:
                progress_hook(
                    {
                        "event": "model_start",
                        "stage": stage_name,
                        "fold": fold_label,
                        "held_out_session": held_out_session,
                        "model": model_name,
                    }
                )
            model = builder()
            started = time.perf_counter()
            model.fit(X_train_eval, y_train_eval.astype(str))
            elapsed = time.perf_counter() - started
            if progress_hook is not None:
                progress_hook(
                    {
                        "event": "model_complete",
                        "stage": stage_name,
                        "fold": fold_label,
                        "held_out_session": held_out_session,
                        "model": model_name,
                        "runtime_seconds": elapsed,
                    }
                )

            if save_models_flag:
                save_model(model, model_path)

        if save_models_flag:
            saved_models.append(
                {
                    "stage": stage_name,
                    "fold": fold_label,
                    "held_out_session": held_out_session,
                    "model": model_name,
                    "model_path": str(model_path),
                    "resumed_from_disk": bool(resumed_from_disk),
                }
            )

        y_pred = pd.Series(model.predict(X_test_eval), index=y_test_eval.index, name=model_name).astype(str)
        proba = safe_predict_proba(model, X_test_eval, all_labels)
        metric_rows.append(
            {
                "stage": stage_name,
                "fold": fold_label,
                "held_out_session": held_out_session,
                "model": model_name,
                "accuracy": accuracy_score(y_test_eval, y_pred),
                "precision_macro": precision_score(y_test_eval, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_test_eval, y_pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y_test_eval, y_pred, average="macro", zero_division=0),
                "precision_weighted": precision_score(y_test_eval, y_pred, average="weighted", zero_division=0),
                "recall_weighted": recall_score(y_test_eval, y_pred, average="weighted", zero_division=0),
                "f1_weighted": f1_score(y_test_eval, y_pred, average="weighted", zero_division=0),
                "precision_micro": precision_score(y_test_eval, y_pred, average="micro", zero_division=0),
                "recall_micro": recall_score(y_test_eval, y_pred, average="micro", zero_division=0),
                "f1_micro": f1_score(y_test_eval, y_pred, average="micro", zero_division=0),
                "runtime_seconds": elapsed,
                "n_test_rows": int(len(y_test_eval)),
                "resumed_from_disk": bool(resumed_from_disk),
            }
        )

        report_dict = classification_report(
            y_test_eval,
            y_pred,
            labels=all_labels,
            target_names=all_labels,
            zero_division=0,
            output_dict=True,
        )
        cm = confusion_matrix(y_test_eval, y_pred, labels=all_labels)
        report_frame = build_classification_report_frame(report_dict, stage_name=stage_name, model_name=model_name)
        binary_frame = compute_binary_classification_stats(cm, labels=all_labels, stage_name=stage_name, model_name=model_name)
        report_frame = report_frame.merge(binary_frame, on=["stage", "model", "label"], how="left")
        report_frame.insert(1, "fold", fold_label)
        report_frame.insert(2, "held_out_session", held_out_session)
        report_frames.append(report_frame)

        conf_frame = build_confusion_matrix_frame(cm, labels=all_labels, stage_name=stage_name, model_name=model_name)
        conf_frame.insert(1, "fold", fold_label)
        conf_frame.insert(2, "held_out_session", held_out_session)
        conf_frames.append(conf_frame)

        conf_norm_frame = build_normalized_confusion_matrix_frame(cm, labels=all_labels, stage_name=stage_name, model_name=model_name)
        conf_norm_frame.insert(1, "fold", fold_label)
        conf_norm_frame.insert(2, "held_out_session", held_out_session)
        conf_norm_frames.append(conf_norm_frame)

        binary_frame.insert(1, "fold", fold_label)
        binary_frame.insert(2, "held_out_session", held_out_session)
        binary_frames.append(binary_frame)

        for row_pos, (row_idx, true_label) in enumerate(y_test_eval.items()):
            payload = {
                "stage": stage_name,
                "fold": fold_label,
                "held_out_session": held_out_session,
                "model": model_name,
                "test_index": int(row_idx),
                "true_label": str(true_label),
                "predicted_label": str(y_pred.loc[row_idx]),
            }
            if proba is not None:
                class_probs = {label: float(proba[row_pos, idx]) for idx, label in enumerate(all_labels)}
                payload["max_probability"] = max(class_probs.values())
                payload["probabilities_json"] = json.dumps(class_probs, sort_keys=True)
            else:
                payload["max_probability"] = None
                payload["probabilities_json"] = None
            prediction_rows.append(payload)

    results_df = pd.DataFrame(metric_rows).sort_values(["stage", "f1_macro", "f1_weighted"], ascending=[True, False, False], ignore_index=True)
    return results_df, report_frames, conf_frames, conf_norm_frames, binary_frames, prediction_rows, saved_models


def fit_final_models(
    *,
    model_builders: dict[str, ModelBuilder],
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    stage_name: str,
    model_dir: Path,
    save_models_flag: bool,
    progress_hook: ProgressHook | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, builder in model_builders.items():
        model_path = final_model_path(model_dir, stage_name, model_name) if save_models_flag else None
        resumed_from_disk = False
        elapsed = float("nan")

        if model_path is not None and model_path.exists():
            try:
                _ = load_model(model_path)
                resumed_from_disk = True
                if progress_hook is not None:
                    progress_hook(
                        {
                            "event": "model_reused",
                            "stage": stage_name,
                            "fold": "full_data",
                            "held_out_session": None,
                            "model": model_name,
                            "model_path": str(model_path),
                        }
                    )
            except Exception as exc:
                print(
                    f"[resume] final_model={model_name} stage={stage_name} "
                    f"status=load_failed error={type(exc).__name__}: {exc} -> refit",
                    flush=True,
                )

        if not resumed_from_disk:
            if progress_hook is not None:
                progress_hook(
                    {
                        "event": "model_start",
                        "stage": stage_name,
                        "fold": "full_data",
                        "held_out_session": None,
                        "model": model_name,
                    }
                )
            model = builder()
            started = time.perf_counter()
            model.fit(X_eval, y_eval.astype(str))
            elapsed = time.perf_counter() - started
            if progress_hook is not None:
                progress_hook(
                    {
                        "event": "model_complete",
                        "stage": stage_name,
                        "fold": "full_data",
                        "held_out_session": None,
                        "model": model_name,
                        "runtime_seconds": elapsed,
                    }
                )
            if model_path is not None:
                save_model(model, model_path)
        rows.append(
            {
                "stage": stage_name,
                "model": model_name,
                "train_rows": int(len(X_eval)),
                "runtime_seconds": elapsed,
                "model_path": str(model_path) if model_path is not None else None,
                "resumed_from_disk": bool(resumed_from_disk),
            }
        )
    return rows


def run_experiment(args: ExperimentArgs) -> dict[str, Any]:
    np.random.seed(args.random_state)
    machine_name = machine_name_from_env()
    variant = variant_name(args.include_xxx)
    model_parallel_jobs, model_parallel_mode = resolve_cpu_parallel_config(machine_name)
    source_path = resolve_feature_source(REPO_ROOT, args.dataset_key)
    out_dir = resolve_augmentation_artifact_dir(
        REPO_ROOT,
        include_xxx=args.include_xxx,
        use_variant_dirs=args.use_variant_dirs,
        run_name=args.run_name,
        dataset_key=args.dataset_key,
    )
    reports_dir = out_dir / "reports"
    figures_dir = out_dir / "figures"
    models_dir = out_dir / "models"
    for directory in (reports_dir, figures_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    env_summary = environment_snapshot(probe_xgboost_gpu=args.enable_gpu_models)
    env_summary.update(
        {
            "machine_name": machine_name,
            "source_path": str(source_path),
            "variant_name": variant,
            "model_parallel_jobs": model_parallel_jobs,
            "model_parallel_mode": model_parallel_mode,
        }
    )
    write_json(out_dir / "environment_summary.json", env_summary)

    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing source CSV for dataset '{args.dataset_key}': {source_path}. "
            "Set SER_FEATURE_SOURCE_ROOT or SER_FEATURE_SOURCE_<DATASET_KEY> before running."
        )

    raw_df = pd.read_csv(source_path, nrows=args.use_nrows)
    target_name = find_column(list(raw_df.columns), args.target_col)
    if target_name is None:
        raise ValueError(f"Target column '{args.target_col}' not found in {source_path}.")
    session_name = find_column(list(raw_df.columns), args.session_col)
    if session_name is None:
        raise ValueError(f"Session column '{args.session_col}' not found in {source_path}.")

    df = filter_feature_frame(
        raw_df,
        target_col=target_name,
        include_xxx=args.include_xxx,
        require_agreement=args.require_agreement,
        excluded_emotions=args.excluded_emotions,
    )
    meta_cols = [column for column in df.columns if is_metadata_column(column)]
    if session_name not in meta_cols:
        meta_cols.append(session_name)
    feature_cols = [column for column in df.columns if column not in meta_cols]

    raw_class_counts = summarize_class_counts(raw_df[target_name], label_name=target_name)
    filtered_class_counts = summarize_class_counts(df[target_name], label_name=target_name)
    save_figure(
        figures_dir / f"{args.dataset_key}_class_distributions.png",
        plot_class_distributions(raw_class_counts, filtered_class_counts, target_col=target_name, variant=variant),
    )

    model_df = df[[target_name, session_name] + feature_cols].copy()
    rows_before = len(model_df)
    model_df = model_df.dropna(subset=[target_name, session_name]).copy()
    usable_feature_cols = [column for column in feature_cols if model_df[column].notna().all()]
    dropped_feature_cols = [column for column in feature_cols if column not in usable_feature_cols]
    model_df = model_df[[target_name, session_name] + usable_feature_cols].dropna(axis=0, how="any").copy()
    model_df[session_name] = model_df[session_name].astype(int)
    rows_after = len(model_df)

    X = model_df[usable_feature_cols].astype(float)
    y = model_df[target_name].astype(str)
    groups = model_df[session_name]
    all_labels = sorted(y.unique())

    if groups.nunique() < 2:
        raise ValueError("Need at least two sessions for leave-one-session-out cross-validation.")

    stage_class_counts_df = build_stage_class_count_table(
        {
            "raw_loaded": raw_df[target_name],
            "filtered": df[target_name],
            "model_ready": model_df[target_name],
        },
        label_name=target_name,
    )

    wandb_run = make_wandb_run(args, dataset_key=args.dataset_key, variant=variant, artifact_dir=out_dir)
    if wandb_run is not None:
        wandb_run.summary["rows_loaded"] = int(len(raw_df))
        wandb_run.summary["rows_model_ready"] = int(rows_after)
        wandb_run.summary["feature_columns_used"] = int(len(usable_feature_cols))
        wandb_run.summary["n_sessions"] = int(groups.nunique())

    model_builders = build_model_builders(
        random_state=args.random_state,
        model_parallel_jobs=model_parallel_jobs,
        enable_gpu_models=args.enable_gpu_models,
        xgb_estimators=args.xgb_estimators,
        xgb_learning_rate=args.xgb_learning_rate,
        xgb_max_depth=args.xgb_max_depth,
        xgb_subsample=args.xgb_subsample,
        xgb_colsample_bytree=args.xgb_colsample_bytree,
        poly_degree=args.poly_degree,
        poly_coef0=args.poly_coef0,
        poly_c=args.poly_c,
    )
    requested_models = list(dict.fromkeys(args.models))
    model_builders = {name: model_builders[name] for name in requested_models}

    logo = LeaveOneGroupOut()
    total_outer_folds = int(groups.nunique())
    model_names = list(model_builders.keys())
    stages_per_fold = ["baseline", "augmented"]
    final_stages = ["baseline_final", "augmented_final"]
    total_model_fits = total_outer_folds * len(stages_per_fold) * len(model_names) + len(final_stages) * len(model_names)
    progress_state = {"completed_model_fits": 0}

    def emit_progress(payload: dict[str, Any]) -> None:
        event = str(payload.get("event", "progress"))
        stage = str(payload.get("stage", "unknown"))
        fold = str(payload.get("fold", "unknown"))
        model = str(payload.get("model", "unknown"))
        held_out_session = payload.get("held_out_session")
        runtime_seconds = payload.get("runtime_seconds")
        model_path = payload.get("model_path")
        if event == "model_start":
            print(
                f"[progress] dataset={args.dataset_key} variant={variant} "
                f"fold={fold} held_out_session={held_out_session} stage={stage} model={model} status=start",
                flush=True,
            )
        elif event == "model_complete":
            progress_state["completed_model_fits"] += 1
            completed = progress_state["completed_model_fits"]
            print(
                f"[progress] dataset={args.dataset_key} variant={variant} "
                f"fold={fold} held_out_session={held_out_session} stage={stage} model={model} "
                f"status=done runtime_seconds={runtime_seconds:.3f} completed_fits={completed}/{total_model_fits}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "progress/completed_model_fits": completed,
                        "progress/total_model_fits": total_model_fits,
                        "progress/current_stage_label": stage,
                        "progress/current_model_label": model,
                        "progress/current_fold_label": fold,
                        "progress/current_held_out_session": held_out_session if held_out_session is not None else -1,
                        f"runtime/{stage}/{model}": float(runtime_seconds),
                    }
                )
        elif event == "model_reused":
            progress_state["completed_model_fits"] += 1
            completed = progress_state["completed_model_fits"]
            print(
                f"[progress] dataset={args.dataset_key} variant={variant} "
                f"fold={fold} held_out_session={held_out_session} stage={stage} model={model} "
                f"status=resume_reuse model_path={model_path} completed_fits={completed}/{total_model_fits}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "progress/completed_model_fits": completed,
                        "progress/total_model_fits": total_model_fits,
                        "progress/current_stage_label": stage,
                        "progress/current_model_label": model,
                        "progress/current_fold_label": fold,
                        "progress/current_held_out_session": held_out_session if held_out_session is not None else -1,
                    }
                )

    fold_metric_frames: list[pd.DataFrame] = []
    report_frames: list[pd.DataFrame] = []
    conf_frames: list[pd.DataFrame] = []
    conf_norm_frames: list[pd.DataFrame] = []
    binary_frames: list[pd.DataFrame] = []
    prediction_rows: list[dict[str, Any]] = []
    saved_models_rows: list[dict[str, Any]] = []
    augmentation_summary_frames: list[pd.DataFrame] = []
    fold_stage_counts_frames: list[pd.DataFrame] = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), start=1):
        held_out_session = int(pd.Series(groups.iloc[test_idx]).iloc[0])
        fold_label = f"session_{held_out_session}"
        checkpoint_dir = checkpoint_dir_for_fold(reports_dir, fold_label)
        checkpoint_meta = checkpoint_dir / "fold_complete.json"
        if checkpoint_meta.exists():
            restored = load_fold_checkpoint(checkpoint_dir=checkpoint_dir, dataset_key=args.dataset_key)
            fold_metric_frames.append(restored["fold_metrics"])
            report_frames.append(restored["classification_reports"])
            conf_frames.append(restored["confusion_matrices"])
            conf_norm_frames.append(restored["confusion_matrices_normalized"])
            binary_frames.append(restored["per_class_accuracy"])
            prediction_rows.extend(restored["fold_predictions"].to_dict(orient="records"))
            saved_models_rows.extend(restored["saved_models"].to_dict(orient="records"))
            augmentation_summary_frames.append(restored["augmentation_summary_by_fold"])
            fold_stage_counts_frames.append(restored["fold_class_counts"])
            progress_state["completed_model_fits"] += len(stages_per_fold) * len(model_names)
            print(
                f"[fold] dataset={args.dataset_key} variant={variant} fold_index={fold_idx}/{total_outer_folds} "
                f"held_out_session={held_out_session} status=resume_skip completed_fits={progress_state['completed_model_fits']}/{total_model_fits}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "progress/current_outer_fold": fold_idx,
                        "progress/total_outer_folds": total_outer_folds,
                        "progress/current_held_out_session": held_out_session,
                        "progress/completed_model_fits": progress_state["completed_model_fits"],
                        "progress/total_model_fits": total_model_fits,
                    }
                )
            continue
        print(
            f"[fold] dataset={args.dataset_key} variant={variant} fold_index={fold_idx}/{total_outer_folds} "
            f"held_out_session={held_out_session} train_rows={len(train_idx)} test_rows={len(test_idx)}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "progress/current_outer_fold": fold_idx,
                    "progress/total_outer_folds": total_outer_folds,
                    "progress/current_held_out_session": held_out_session,
                    "progress/train_rows": int(len(train_idx)),
                    "progress/test_rows": int(len(test_idx)),
                }
            )
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=False).rename(columns={"index": "original_index"})
        y_test = y.iloc[test_idx].reset_index(drop=False).rename(columns={"index": "original_index", target_name: target_name})
        X_test_features = X.iloc[test_idx].reset_index(drop=True)
        y_test_labels = y.iloc[test_idx].reset_index(drop=True)

        fold_stage_counts = build_stage_class_count_table(
            {
                "train_reference": y_train,
                "test_reference": y_test_labels,
            },
            label_name=target_name,
        )
        fold_stage_counts.insert(0, "fold", fold_label)
        fold_stage_counts.insert(1, "held_out_session", held_out_session)
        fold_stage_counts_frames.append(fold_stage_counts)

        baseline_results, baseline_reports, baseline_confs, baseline_confs_norm, baseline_binary, baseline_predictions, baseline_saved = fit_models_for_fold(
            model_builders=model_builders,
            X_train_eval=X_train,
            y_train_eval=y_train,
            X_test_eval=X_test_features,
            y_test_eval=y_test_labels,
            stage_name="baseline",
            fold_label=fold_label,
            held_out_session=held_out_session,
            all_labels=all_labels,
            models_dir=models_dir,
            save_models_flag=args.save_models,
            progress_hook=emit_progress,
        )
        fold_metric_frames.append(baseline_results)
        report_frames.extend(baseline_reports)
        conf_frames.extend(baseline_confs)
        conf_norm_frames.extend(baseline_confs_norm)
        binary_frames.extend(baseline_binary)
        prediction_rows.extend(baseline_predictions)
        saved_models_rows.extend(baseline_saved)

        X_train_aug, y_train_aug, synthetic_meta_df, augmentation_summary_df = mean_std_oversample_training_frame(
            X_train,
            y_train,
            target_per_class=args.target_per_class,
            group_size=args.group_size,
            random_state=args.augmentation_random_state,
            sample_with_replacement=True,
        )
        augmentation_summary_df = augmentation_summary_df.copy()
        augmentation_summary_df.insert(0, "fold", fold_label)
        augmentation_summary_df.insert(1, "held_out_session", held_out_session)
        augmentation_summary_frames.append(augmentation_summary_df)

        synthetic_only_labels = synthetic_meta_df[target_name] if target_name in synthetic_meta_df.columns else pd.Series(dtype=str)
        fold_aug_counts = build_stage_class_count_table(
            {
                "synthetic_only": synthetic_only_labels,
                "train_augmented": y_train_aug,
            },
            label_name=target_name,
        )
        fold_aug_counts.insert(0, "fold", fold_label)
        fold_aug_counts.insert(1, "held_out_session", held_out_session)
        fold_stage_counts_frames.append(fold_aug_counts)

        augmented_results, augmented_reports, augmented_confs, augmented_confs_norm, augmented_binary, augmented_predictions, augmented_saved = fit_models_for_fold(
            model_builders=model_builders,
            X_train_eval=X_train_aug,
            y_train_eval=y_train_aug,
            X_test_eval=X_test_features,
            y_test_eval=y_test_labels,
            stage_name="augmented",
            fold_label=fold_label,
            held_out_session=held_out_session,
            all_labels=all_labels,
            models_dir=models_dir,
            save_models_flag=args.save_models,
            progress_hook=emit_progress,
        )
        fold_metric_frames.append(augmented_results)
        report_frames.extend(augmented_reports)
        conf_frames.extend(augmented_confs)
        conf_norm_frames.extend(augmented_confs_norm)
        binary_frames.extend(augmented_binary)
        prediction_rows.extend(augmented_predictions)
        saved_models_rows.extend(augmented_saved)

        if wandb_run is not None:
            best_baseline = baseline_results.sort_values(["f1_macro", "f1_weighted"], ascending=False).iloc[0]
            best_augmented = augmented_results.sort_values(["f1_macro", "f1_weighted"], ascending=False).iloc[0]
            wandb_run.log(
                {
                    "fold": fold_idx,
                    "held_out_session": held_out_session,
                    "baseline/best_accuracy": float(best_baseline["accuracy"]),
                    "baseline/best_f1_macro": float(best_baseline["f1_macro"]),
                    "augmented/best_accuracy": float(best_augmented["accuracy"]),
                    "augmented/best_f1_macro": float(best_augmented["f1_macro"]),
                }
            )

        if args.save_augmented_matrices:
            fold_data_dir = reports_dir / "fold_datasets" / fold_label
            fold_data_dir.mkdir(parents=True, exist_ok=True)
            X_train_aug.to_csv(fold_data_dir / f"{args.dataset_key}_X_train_augmented.csv", index=False)
            y_train_aug.to_frame(name=target_name).to_csv(fold_data_dir / f"{args.dataset_key}_y_train_augmented.csv", index=False)
            X_test_features.to_csv(fold_data_dir / f"{args.dataset_key}_X_test_reference.csv", index=False)
            y_test_labels.to_frame(name=target_name).to_csv(fold_data_dir / f"{args.dataset_key}_y_test_reference.csv", index=False)
            synthetic_meta_df.to_csv(fold_data_dir / f"{args.dataset_key}_synthetic_metadata.csv", index=False)

        save_fold_checkpoint(
            checkpoint_dir=checkpoint_dir,
            dataset_key=args.dataset_key,
            fold_label=fold_label,
            held_out_session=held_out_session,
            baseline_results=baseline_results,
            augmented_results=augmented_results,
            baseline_reports=baseline_reports,
            augmented_reports=augmented_reports,
            baseline_confs=baseline_confs,
            augmented_confs=augmented_confs,
            baseline_confs_norm=baseline_confs_norm,
            augmented_confs_norm=augmented_confs_norm,
            baseline_binary=baseline_binary,
            augmented_binary=augmented_binary,
            baseline_predictions=baseline_predictions,
            augmented_predictions=augmented_predictions,
            baseline_saved=baseline_saved,
            augmented_saved=augmented_saved,
            augmentation_summary_df=augmentation_summary_df,
            fold_stage_counts=[fold_stage_counts, fold_aug_counts],
        )

    fold_metrics_df = pd.concat(fold_metric_frames, ignore_index=True)
    classification_reports_df = pd.concat(report_frames, ignore_index=True)
    confusion_matrices_df = pd.concat(conf_frames, ignore_index=True)
    confusion_norm_df = pd.concat(conf_norm_frames, ignore_index=True)
    binary_class_metrics_df = pd.concat(binary_frames, ignore_index=True)
    prediction_df = pd.DataFrame(prediction_rows)
    saved_models_df = pd.DataFrame(saved_models_rows)
    augmentation_summary_all_df = pd.concat(augmentation_summary_frames, ignore_index=True) if augmentation_summary_frames else pd.DataFrame()
    fold_stage_counts_df = pd.concat(fold_stage_counts_frames, ignore_index=True) if fold_stage_counts_frames else pd.DataFrame()

    metric_columns = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "precision_micro",
        "recall_micro",
        "f1_micro",
        "runtime_seconds",
    ]
    aggregate_metrics_df = (
        fold_metrics_df.groupby(["stage", "model"], as_index=False)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate_metrics_df.columns = [
        "_".join(str(part) for part in col if part != "").strip("_") for col in aggregate_metrics_df.columns.to_flat_index()
    ]
    aggregate_metrics_df = aggregate_metrics_df.rename(columns={"stage": "stage", "model": "model"})

    best_baseline_row = aggregate_metrics_df[aggregate_metrics_df["stage"] == "baseline"].sort_values(
        ["f1_macro_mean", "f1_weighted_mean"], ascending=False
    ).iloc[0]
    best_augmented_row = aggregate_metrics_df[aggregate_metrics_df["stage"] == "augmented"].sort_values(
        ["f1_macro_mean", "f1_weighted_mean"], ascending=False
    ).iloc[0]
    best_baseline_model = str(best_baseline_row["model"])
    best_augmented_model = str(best_augmented_row["model"])

    pooled_best_baseline = prediction_df[(prediction_df["stage"] == "baseline") & (prediction_df["model"] == best_baseline_model)].copy()
    pooled_best_augmented = prediction_df[(prediction_df["stage"] == "augmented") & (prediction_df["model"] == best_augmented_model)].copy()
    save_figure(
        figures_dir / f"{args.dataset_key}_baseline_confusion_best.png",
        plot_confusion_heatmaps(
            pooled_best_baseline["true_label"],
            pooled_best_baseline["predicted_label"],
            labels=all_labels,
            title_prefix=f"Baseline {best_baseline_model}",
        ),
    )
    save_figure(
        figures_dir / f"{args.dataset_key}_augmented_confusion_best.png",
        plot_confusion_heatmaps(
            pooled_best_augmented["true_label"],
            pooled_best_augmented["predicted_label"],
            labels=all_labels,
            title_prefix=f"Augmented {best_augmented_model}",
        ),
    )
    save_figure(
        figures_dir / f"{args.dataset_key}_macro_f1_summary.png",
        plot_fold_metric_summary(aggregate_metrics_df, metric_name="f1_macro_mean", title="Mean Macro F1 Across Sessions"),
    )
    save_figure(
        figures_dir / f"{args.dataset_key}_accuracy_by_session.png",
        plot_session_metric_heatmap(fold_metrics_df, metric_name="accuracy"),
    )

    final_model_rows = []
    final_model_rows.extend(
        fit_final_models(
            model_builders=model_builders,
            X_eval=X,
            y_eval=y,
            stage_name="baseline_final",
            model_dir=models_dir / "final",
            save_models_flag=args.save_models,
            progress_hook=emit_progress,
        )
    )
    X_full_aug, y_full_aug, synthetic_meta_full_df, augmentation_summary_full_df = mean_std_oversample_training_frame(
        X,
        y,
        target_per_class=args.target_per_class,
        group_size=args.group_size,
        random_state=args.augmentation_random_state,
        sample_with_replacement=True,
    )
    final_model_rows.extend(
        fit_final_models(
            model_builders=model_builders,
            X_eval=X_full_aug,
            y_eval=y_full_aug,
            stage_name="augmented_final",
            model_dir=models_dir / "final",
            save_models_flag=args.save_models,
            progress_hook=emit_progress,
        )
    )
    final_models_df = pd.DataFrame(final_model_rows)

    reports_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics_df.to_csv(reports_dir / f"{args.dataset_key}_fold_metrics.csv", index=False)
    aggregate_metrics_df.to_csv(reports_dir / f"{args.dataset_key}_aggregate_metrics.csv", index=False)
    classification_reports_df.to_csv(reports_dir / f"{args.dataset_key}_classification_reports.csv", index=False)
    confusion_matrices_df.to_csv(reports_dir / f"{args.dataset_key}_confusion_matrices.csv", index=False)
    confusion_norm_df.to_csv(reports_dir / f"{args.dataset_key}_confusion_matrices_normalized.csv", index=False)
    binary_class_metrics_df.to_csv(reports_dir / f"{args.dataset_key}_per_class_accuracy.csv", index=False)
    prediction_df.to_csv(reports_dir / f"{args.dataset_key}_fold_predictions.csv", index=False)
    saved_models_df.to_csv(reports_dir / f"{args.dataset_key}_saved_models.csv", index=False)
    final_models_df.to_csv(reports_dir / f"{args.dataset_key}_final_models.csv", index=False)
    stage_class_counts_df.to_csv(reports_dir / f"{args.dataset_key}_class_counts.csv", index=False)
    fold_stage_counts_df.to_csv(reports_dir / f"{args.dataset_key}_fold_class_counts.csv", index=False)
    augmentation_summary_all_df.to_csv(reports_dir / f"{args.dataset_key}_augmentation_summary_by_fold.csv", index=False)
    augmentation_summary_full_df.to_csv(reports_dir / f"{args.dataset_key}_augmentation_summary_full_data.csv", index=False)
    synthetic_meta_full_df.to_csv(reports_dir / f"{args.dataset_key}_synthetic_metadata_full_data.csv", index=False)

    run_meta = {
        "dataset_key": args.dataset_key,
        "run_name": args.run_name,
        "variant_name": variant,
        "source_path": str(source_path),
        "artifact_dir": str(out_dir),
        "reports_dir": str(reports_dir),
        "figures_dir": str(figures_dir),
        "models_dir": str(models_dir),
        "environment_summary": env_summary,
        "target_col": target_name,
        "session_col": session_name,
        "include_xxx": args.include_xxx,
        "require_agreement": args.require_agreement,
        "excluded_emotions": args.excluded_emotions,
        "rows_loaded": int(len(raw_df)),
        "rows_after_filter": int(len(df)),
        "rows_used": int(len(model_df)),
        "rows_before_cleaning": int(rows_before),
        "rows_after_cleaning": int(rows_after),
        "feature_columns_used": int(len(usable_feature_cols)),
        "dropped_feature_columns": dropped_feature_cols,
        "classes": all_labels,
        "sessions": sorted(pd.unique(groups).tolist()),
        "n_folds": int(groups.nunique()),
        "model_parallel_jobs": model_parallel_jobs,
        "model_parallel_mode": model_parallel_mode,
        "augmentation_method": DEFAULT_AUGMENTATION_METHOD,
        "augmentation_target_per_class": args.target_per_class,
        "augmentation_group_size": args.group_size,
        "augmentation_random_state": args.augmentation_random_state,
        "best_baseline_model": best_baseline_model,
        "best_augmented_model": best_augmented_model,
        "best_baseline_metrics": best_baseline_row.to_dict(),
        "best_augmented_metrics": best_augmented_row.to_dict(),
        "artifact_files": {
            "fold_metrics": str(reports_dir / f"{args.dataset_key}_fold_metrics.csv"),
            "aggregate_metrics": str(reports_dir / f"{args.dataset_key}_aggregate_metrics.csv"),
            "classification_reports": str(reports_dir / f"{args.dataset_key}_classification_reports.csv"),
            "confusion_matrices": str(reports_dir / f"{args.dataset_key}_confusion_matrices.csv"),
            "confusion_matrices_normalized": str(reports_dir / f"{args.dataset_key}_confusion_matrices_normalized.csv"),
            "per_class_accuracy": str(reports_dir / f"{args.dataset_key}_per_class_accuracy.csv"),
            "fold_predictions": str(reports_dir / f"{args.dataset_key}_fold_predictions.csv"),
            "saved_models": str(reports_dir / f"{args.dataset_key}_saved_models.csv"),
            "final_models": str(reports_dir / f"{args.dataset_key}_final_models.csv"),
            "class_counts": str(reports_dir / f"{args.dataset_key}_class_counts.csv"),
            "fold_class_counts": str(reports_dir / f"{args.dataset_key}_fold_class_counts.csv"),
            "augmentation_summary_by_fold": str(reports_dir / f"{args.dataset_key}_augmentation_summary_by_fold.csv"),
            "augmentation_summary_full_data": str(reports_dir / f"{args.dataset_key}_augmentation_summary_full_data.csv"),
            "synthetic_metadata_full_data": str(reports_dir / f"{args.dataset_key}_synthetic_metadata_full_data.csv"),
        },
    }
    write_json(out_dir / f"{args.dataset_key}_run_metadata.json", run_meta)

    if wandb_run is not None:
        import wandb

        wandb_run.log(
            {
                "fold_metrics": wandb.Table(dataframe=fold_metrics_df),
                "aggregate_metrics": wandb.Table(dataframe=aggregate_metrics_df),
                "classification_reports": wandb.Table(dataframe=classification_reports_df),
                "per_class_accuracy": wandb.Table(dataframe=binary_class_metrics_df),
                "saved_models": wandb.Table(dataframe=saved_models_df),
                "final_models": wandb.Table(dataframe=final_models_df),
            }
        )
        for image_path in sorted(figures_dir.glob("*.png")):
            wandb_run.log({image_path.stem: wandb.Image(str(image_path))})
        wandb_run.summary["best_baseline_model"] = best_baseline_model
        wandb_run.summary["best_augmented_model"] = best_augmented_model
        wandb_run.summary["best_baseline_f1_macro_mean"] = float(best_baseline_row["f1_macro_mean"])
        wandb_run.summary["best_augmented_f1_macro_mean"] = float(best_augmented_row["f1_macro_mean"])
        wandb_run.finish()

    return {
        "dataset_key": args.dataset_key,
        "variant": variant,
        "status": "ok",
        "artifact_dir": str(out_dir),
        "best_baseline_model": best_baseline_model,
        "best_augmented_model": best_augmented_model,
        "baseline_best_f1_weighted": float(best_baseline_row["f1_weighted_mean"]),
        "baseline_best_f1_macro": float(best_baseline_row["f1_macro_mean"]),
        "augmented_best_f1_weighted": float(best_augmented_row["f1_weighted_mean"]),
        "augmented_best_f1_macro": float(best_augmented_row["f1_macro_mean"]),
        "baseline_best_accuracy": float(best_baseline_row["accuracy_mean"]),
        "augmented_best_accuracy": float(best_augmented_row["accuracy_mean"]),
        "n_folds": int(groups.nunique()),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    try:
        result = run_experiment(args)
        result["seconds"] = round(time.perf_counter() - started, 3)
        print(json.dumps(result))
        return 0
    except Exception as exc:  # noqa: BLE001
        failure = {
            "dataset_key": args.dataset_key,
            "variant": variant_name(args.include_xxx),
            "status": "failed",
            "seconds": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(failure))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
