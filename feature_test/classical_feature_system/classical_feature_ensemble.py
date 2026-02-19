"""Classical feature-based ensemble pipeline for SER.

This script implements the "feature-based classical model system" branch:
- Load multiple feature sets (MFCC, TF-IDF, BERT, prosody, etc.).
- Train several classical models per feature set.
- Select the best model per feature set on validation macro-F1.
- Fuse per-feature-set probabilities with weighted late fusion.
- Evaluate and save metrics/artifacts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

METADATA_COLS = {
    "path",
    "session",
    "method",
    "gender",
    "emotion",
    "n_annotators",
    "agreement",
    "utt_id",
    "text",
    "split",
}


@dataclass(frozen=True)
class FeatureSetConfig:
    key: str
    csv_path: Path
    selected_features_json: Path | None = None
    id_col: str = "path"
    target_col: str = "emotion"


def resolve_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").exists():
        return cwd
    for parent in cwd.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find project root with pyproject.toml")


def default_feature_sets(repo_root: Path) -> dict[str, FeatureSetConfig]:
    artifacts = repo_root / "feature_selection" / "filter" / "select_k_best_anova" / "artifacts"
    return {
        "mfcc_normalized": FeatureSetConfig(
            key="mfcc_normalized",
            csv_path=repo_root / "extracted_features" / "mfcc" / "mfcc_features_normalized.csv",
            selected_features_json=artifacts / "mfcc_normalized_selected_features.json",
        ),
        "bert": FeatureSetConfig(
            key="bert",
            csv_path=repo_root / "extracted_features" / "text" / "bert_embeddings.csv",
            selected_features_json=artifacts / "bert_selected_features.json",
        ),
        "tfidf": FeatureSetConfig(
            key="tfidf",
            csv_path=repo_root / "extracted_features" / "text" / "tfidf_features.csv",
            selected_features_json=artifacts / "tfidf_selected_features.json",
        ),
        "prosody_energy": FeatureSetConfig(
            key="prosody_energy",
            csv_path=repo_root / "extracted_features" / "prosody_energy" / "prosody_energy_features.csv",
            selected_features_json=artifacts / "prosody_energy_selected_features.json",
        ),
        "prosody_pitch": FeatureSetConfig(
            key="prosody_pitch",
            csv_path=repo_root / "extracted_features" / "prosody_pitch" / "prosody_pitch_features.csv",
            selected_features_json=artifacts / "prosody_pitch_selected_features.json",
        ),
    }


def _load_selected_features(path: Path | None) -> list[str] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return [str(x) for x in data]


def _header_columns(csv_path: Path) -> list[str]:
    return list(pd.read_csv(csv_path, nrows=0).columns)


def load_feature_frame(
    cfg: FeatureSetConfig,
    *,
    use_selected: bool,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    if not cfg.csv_path.exists():
        raise FileNotFoundError(cfg.csv_path)

    cols_all = _header_columns(cfg.csv_path)
    if cfg.id_col not in cols_all or cfg.target_col not in cols_all:
        raise ValueError(f"{cfg.key}: missing required columns {cfg.id_col}/{cfg.target_col}")

    selected = _load_selected_features(cfg.selected_features_json) if use_selected else None
    if selected:
        feature_cols = [c for c in selected if c in cols_all and c not in METADATA_COLS]
    else:
        feature_cols = [c for c in cols_all if c not in METADATA_COLS]
    if not feature_cols:
        raise ValueError(f"{cfg.key}: no usable feature columns found")

    usecols = [cfg.id_col, cfg.target_col] + feature_cols
    df = pd.read_csv(cfg.csv_path, usecols=usecols)
    df = df.dropna(subset=[cfg.id_col, cfg.target_col]).copy()

    # If duplicates exist, keep first to guarantee 1 row per id for merge.
    df = df.drop_duplicates(subset=[cfg.id_col], keep="first")

    renamed = {c: f"{cfg.key}__{c}" for c in feature_cols}
    df = df.rename(columns=renamed)
    prefixed = list(renamed.values())

    if verbose:
        sel_info = "ANOVA-selected" if selected else "all"
        print(
            f"[{cfg.key}] rows={len(df)} features={len(prefixed)} source={sel_info} "
            f"path={cfg.csv_path}"
        )
    return df, prefixed


def build_model_zoo(random_state: int) -> dict[str, Any]:
    return {
        "logreg": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=None,
                    ),
                ),
            ]
        ),
        "svm_rbf": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=3.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gb": HistGradientBoostingClassifier(
            random_state=random_state,
            max_iter=300,
            learning_rate=0.08,
        ),
        "mlp": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 128),
                        activation="relu",
                        alpha=1e-4,
                        max_iter=250,
                        early_stopping=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def proba_or_softmax(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)
    if not hasattr(model, "decision_function"):
        raise AttributeError("Model supports neither predict_proba nor decision_function")
    scores = model.decision_function(x)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    return softmax(scores, axis=1)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classical feature-based ensemble for SER")
    parser.add_argument(
        "--feature-sets",
        default="mfcc_normalized,bert,tfidf,prosody_energy,prosody_pitch",
        help="Comma-separated feature set keys",
    )
    parser.add_argument(
        "--models",
        default="logreg,svm_rbf,random_forest,hist_gb,mlp",
        help="Comma-separated model names from the model zoo",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--use-selected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ANOVA-selected features when JSON is available",
    )
    parser.add_argument(
        "--out-dir",
        default="feature_test/TreeBased/artifacts/classical_ensemble",
        help="Output directory for reports/artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cfg = default_feature_sets(repo_root)
    selected_keys = [k.strip() for k in args.feature_sets.split(",") if k.strip()]
    missing = [k for k in selected_keys if k not in all_cfg]
    if missing:
        raise ValueError(f"Unknown feature set keys: {missing}. Available: {sorted(all_cfg)}")

    # Load and merge feature-set tables.
    merged: pd.DataFrame | None = None
    feature_cols_by_set: dict[str, list[str]] = {}
    id_col = "path"
    target_col = "emotion"

    print(f"Using feature sets: {selected_keys}")
    for key in selected_keys:
        df_i, prefixed_cols = load_feature_frame(all_cfg[key], use_selected=args.use_selected)
        feature_cols_by_set[key] = prefixed_cols
        if merged is None:
            merged = df_i
        else:
            merged = merged.merge(df_i, on=[id_col, target_col], how="inner")

    if merged is None or merged.empty:
        raise ValueError("Merged dataset is empty")

    # Drop any rows that still contain NaN in feature columns.
    all_feature_cols = [c for cols in feature_cols_by_set.values() for c in cols]
    merged = merged.dropna(subset=all_feature_cols).copy()
    if merged.empty:
        raise ValueError("Merged dataset is empty after dropna")

    print(
        f"Merged rows={len(merged)} total_features={len(all_feature_cols)} "
        f"classes={merged[target_col].nunique()}"
    )

    # Global split (same samples for every feature-set branch).
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(merged[target_col].astype(str).to_numpy())
    idx = np.arange(len(merged))

    idx_train, idx_test = train_test_split(
        idx,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    y_train = y[idx_train]
    y_test = y[idx_test]

    idx_tr, idx_val = train_test_split(
        idx_train,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=y_train,
    )
    y_tr = y[idx_tr]
    y_val = y[idx_val]

    model_zoo = build_model_zoo(args.random_state)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown_models = [m for m in model_names if m not in model_zoo]
    if unknown_models:
        raise ValueError(f"Unknown model names: {unknown_models}. Available: {sorted(model_zoo)}")

    # Track full experiment table and selected models per feature set.
    rows: list[dict[str, Any]] = []
    best_by_set: dict[str, dict[str, Any]] = {}
    test_proba_by_set: dict[str, np.ndarray] = {}
    test_pred_by_set: dict[str, np.ndarray] = {}

    for set_key in selected_keys:
        cols = feature_cols_by_set[set_key]
        x_all = merged[cols].to_numpy(dtype=np.float64, copy=False)
        x_tr = x_all[idx_tr]
        x_val = x_all[idx_val]
        x_train = x_all[idx_train]
        x_test = x_all[idx_test]

        print(f"\n=== Feature set: {set_key} ({x_all.shape[1]} dims) ===")

        best_val_f1 = -1.0
        best_model_name = None

        for model_name in model_names:
            model = model_zoo[model_name]
            model.fit(x_tr, y_tr)
            pred_val = model.predict(x_val)
            metrics_val = evaluate(y_val, pred_val)
            rows.append(
                {
                    "feature_set": set_key,
                    "model": model_name,
                    "split": "val",
                    **metrics_val,
                }
            )
            print(
                f"[{set_key}] {model_name:<14} val_f1_macro={metrics_val['f1_macro']:.4f} "
                f"val_acc={metrics_val['accuracy']:.4f}"
            )
            if metrics_val["f1_macro"] > best_val_f1:
                best_val_f1 = metrics_val["f1_macro"]
                best_model_name = model_name

        assert best_model_name is not None
        print(f"[{set_key}] best_model={best_model_name} best_val_f1_macro={best_val_f1:.4f}")

        # Refit best model on full train and evaluate on held-out test.
        best_model = model_zoo[best_model_name]
        best_model.fit(x_train, y_train)
        pred_test = best_model.predict(x_test)
        proba_test = proba_or_softmax(best_model, x_test)
        metrics_test = evaluate(y_test, pred_test)

        rows.append(
            {
                "feature_set": set_key,
                "model": best_model_name,
                "split": "test_best",
                **metrics_test,
            }
        )
        print(
            f"[{set_key}] test_best_f1_macro={metrics_test['f1_macro']:.4f} "
            f"test_best_acc={metrics_test['accuracy']:.4f}"
        )

        best_by_set[set_key] = {
            "model_name": best_model_name,
            "val_f1_macro": float(best_val_f1),
            "test_metrics": metrics_test,
        }
        test_proba_by_set[set_key] = proba_test
        test_pred_by_set[set_key] = pred_test

    # Ensemble module: weighted late fusion on per-set probabilities.
    weights = np.array([max(best_by_set[k]["val_f1_macro"], 1e-8) for k in selected_keys], dtype=float)
    weights = weights / weights.sum()

    fused_proba = np.zeros_like(next(iter(test_proba_by_set.values())))
    for w, k in zip(weights, selected_keys, strict=True):
        fused_proba += w * test_proba_by_set[k]
    pred_fused = np.argmax(fused_proba, axis=1)
    metrics_fused = evaluate(y_test, pred_fused)

    rows.append(
        {
            "feature_set": "late_fusion",
            "model": "weighted_avg_proba",
            "split": "test",
            **metrics_fused,
        }
    )

    print("\n=== Ensemble module (late fusion) ===")
    print(
        f"weights={{{', '.join(f'{k}:{w:.3f}' for k, w in zip(selected_keys, weights, strict=True))}}}"
    )
    print(
        f"test_fused_f1_macro={metrics_fused['f1_macro']:.4f} "
        f"test_fused_acc={metrics_fused['accuracy']:.4f}"
    )
    print("\nClassification report (late fusion):")
    print(classification_report(y_test, pred_fused, target_names=label_enc.classes_, zero_division=0))

    # Save artifacts.
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "metrics_table.csv", index=False)

    test_index = merged.iloc[idx_test].reset_index(drop=True)
    pred_out = pd.DataFrame(
        {
            "path": test_index[id_col].astype(str),
            "emotion_true": label_enc.inverse_transform(y_test),
            "emotion_pred_late_fusion": label_enc.inverse_transform(pred_fused),
        }
    )
    for k in selected_keys:
        pred_out[f"emotion_pred_{k}"] = label_enc.inverse_transform(test_pred_by_set[k])
    pred_out.to_csv(out_dir / "test_predictions.csv", index=False)

    summary = {
        "feature_sets": selected_keys,
        "models_considered": model_names,
        "use_selected_features": bool(args.use_selected),
        "rows_after_merge": int(len(merged)),
        "train_rows": int(len(idx_train)),
        "test_rows": int(len(idx_test)),
        "classes": [str(c) for c in label_enc.classes_],
        "best_by_feature_set": best_by_set,
        "late_fusion_weights": {k: float(w) for k, w in zip(selected_keys, weights, strict=True)},
        "late_fusion_test_metrics": metrics_fused,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nSaved artifacts to: {out_dir}")
    print("- metrics_table.csv")
    print("- test_predictions.csv")
    print("- summary.json")


if __name__ == "__main__":
    main()

