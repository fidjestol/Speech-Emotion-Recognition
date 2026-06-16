from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TEXT_COLUMN = "text"
TARGET_COLUMN = "emotion"
KEY_COLUMN = "path"
BASE_TABLE = Path("extracted_features/text/bert_embeddings.csv")
ANOVA_DIR = Path("feature_selection/anova/select_k_best_anova/artifacts")

METADATA_COLUMNS = {
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
    "duration_s",
}

FAMILY_SOURCES: dict[str, Path] = {
    "bert": Path("extracted_features/text/bert_embeddings.csv"),
    "mfcc_normalized": Path("extracted_features/mfcc/mfcc_features_normalized.csv"),
    "prosody_energy": Path("extracted_features/prosody_energy/prosody_energy_features.csv"),
    "prosody_pitch": Path("extracted_features/prosody_pitch/prosody_pitch_features.csv"),
    "rhythm_pauses": Path("extracted_features/rhythm_pauses/rhythm_pauses_features.csv"),
    "spectral_shape": Path("extracted_features/spectral_shape/spectral_shape_features.csv"),
    "tonality": Path("extracted_features/tonality/tonality_features.csv"),
    "voice_quality": Path("extracted_features/voice_quality/voice_quality_features.csv"),
    "ssl_hubert": Path("extracted_features/ssl_embeddings/ssl_embeddings_hubert_features.csv"),
    "ssl_wav2vec": Path("extracted_features/ssl_embeddings/ssl_embeddings_wav2vec_features.csv"),
}


@dataclass(frozen=True)
class FamilySpec:
    name: str
    selected: bool = False


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    modality: str
    tier: str
    text_vectorizer: str | None
    families: tuple[FamilySpec, ...] = field(default_factory=tuple)
    classifier: str = "logreg_low"
    description: str = ""


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="text_low",
        modality="text",
        tier="low",
        text_vectorizer="word_tfidf_low",
        classifier="logreg_low",
        description="Raw transcript TF-IDF unigrams with a small balanced logistic-regression classifier.",
    ),
    ModelSpec(
        model_id="text_medium",
        modality="text",
        tier="medium",
        text_vectorizer="word_tfidf_medium",
        families=(FamilySpec("bert", selected=True),),
        classifier="logreg_medium",
        description="Raw transcript TF-IDF plus ANOVA-selected BERT text embeddings.",
    ),
    ModelSpec(
        model_id="text_high",
        modality="text",
        tier="high",
        text_vectorizer="word_char_tfidf_high",
        families=(FamilySpec("bert", selected=False),),
        classifier="sgd_high",
        description="Raw transcript word/character TF-IDF plus full BERT text embeddings.",
    ),
    ModelSpec(
        model_id="audio_low",
        modality="audio",
        tier="low",
        text_vectorizer=None,
        families=(
            FamilySpec("prosody_energy", selected=False),
            FamilySpec("prosody_pitch", selected=True),
            FamilySpec("rhythm_pauses", selected=False),
        ),
        classifier="logreg_low",
        description="Lightweight prosody, pitch, rhythm, and pause descriptors only.",
    ),
    ModelSpec(
        model_id="audio_medium",
        modality="audio",
        tier="medium",
        text_vectorizer=None,
        families=(
            FamilySpec("prosody_energy", selected=False),
            FamilySpec("prosody_pitch", selected=True),
            FamilySpec("rhythm_pauses", selected=False),
            FamilySpec("spectral_shape", selected=False),
            FamilySpec("tonality", selected=False),
            FamilySpec("voice_quality", selected=False),
            FamilySpec("mfcc_normalized", selected=True),
        ),
        classifier="logreg_medium",
        description="Handcrafted acoustic descriptors plus selected normalized MFCC features.",
    ),
    ModelSpec(
        model_id="audio_high",
        modality="audio",
        tier="high",
        text_vectorizer=None,
        families=(
            FamilySpec("prosody_energy", selected=False),
            FamilySpec("prosody_pitch", selected=True),
            FamilySpec("rhythm_pauses", selected=False),
            FamilySpec("spectral_shape", selected=False),
            FamilySpec("tonality", selected=False),
            FamilySpec("voice_quality", selected=False),
            FamilySpec("mfcc_normalized", selected=True),
            FamilySpec("ssl_hubert", selected=True),
            FamilySpec("ssl_wav2vec", selected=True),
        ),
        classifier="sgd_high",
        description="Handcrafted acoustic descriptors, selected MFCCs, and selected SSL audio embeddings.",
    ),
    ModelSpec(
        model_id="audio_text_low",
        modality="audio_text",
        tier="low",
        text_vectorizer="word_tfidf_low",
        families=(
            FamilySpec("prosody_energy", selected=False),
            FamilySpec("prosody_pitch", selected=True),
            FamilySpec("rhythm_pauses", selected=False),
        ),
        classifier="logreg_low",
        description="Raw transcript TF-IDF with lightweight prosody, pitch, rhythm, and pause descriptors.",
    ),
    ModelSpec(
        model_id="audio_text_medium",
        modality="audio_text",
        tier="medium",
        text_vectorizer="word_tfidf_medium",
        families=(
            FamilySpec("bert", selected=True),
            FamilySpec("prosody_energy", selected=False),
            FamilySpec("prosody_pitch", selected=True),
            FamilySpec("rhythm_pauses", selected=False),
            FamilySpec("spectral_shape", selected=False),
            FamilySpec("tonality", selected=False),
            FamilySpec("voice_quality", selected=False),
            FamilySpec("mfcc_normalized", selected=True),
        ),
        classifier="logreg_medium",
        description="Text TF-IDF and selected BERT embeddings with medium acoustic feature coverage.",
    ),
    ModelSpec(
        model_id="audio_text_high",
        modality="audio_text",
        tier="high",
        text_vectorizer="word_char_tfidf_high",
        families=(
            FamilySpec("bert", selected=False),
            FamilySpec("prosody_energy", selected=False),
            FamilySpec("prosody_pitch", selected=True),
            FamilySpec("rhythm_pauses", selected=False),
            FamilySpec("spectral_shape", selected=False),
            FamilySpec("tonality", selected=False),
            FamilySpec("voice_quality", selected=False),
            FamilySpec("mfcc_normalized", selected=True),
            FamilySpec("ssl_hubert", selected=True),
            FamilySpec("ssl_wav2vec", selected=True),
        ),
        classifier="sgd_high",
        description="Fullest text and audio package: word/character TF-IDF, BERT, handcrafted audio, MFCC, and SSL features.",
    ),
)


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists():
            return path
    raise FileNotFoundError("Could not locate repository root with pyproject.toml.")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def selected_feature_path(repo_root: Path, family: str) -> Path:
    return repo_root / ANOVA_DIR / f"{family}_selected_features.json"


def family_feature_columns(repo_root: Path, family: str, selected: bool) -> list[str]:
    csv_path = repo_root / FAMILY_SOURCES[family]
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing feature table for {family}: {csv_path}")

    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    available_features = [col for col in columns if col not in METADATA_COLUMNS]

    if not selected:
        return available_features

    json_path = selected_feature_path(repo_root, family)
    if not json_path.exists():
        raise FileNotFoundError(f"Missing selected-feature list for {family}: {json_path}")

    requested = [str(col) for col in read_json(json_path)]
    selected_cols = [col for col in requested if col in available_features]
    if not selected_cols:
        raise ValueError(f"No selected features from {json_path} are present in {csv_path}")
    return selected_cols


def load_base_frame(repo_root: Path, include_xxx: bool) -> pd.DataFrame:
    base_path = repo_root / BASE_TABLE
    if not base_path.exists():
        raise FileNotFoundError(f"Missing base table: {base_path}")

    df = pd.read_csv(base_path, usecols=[KEY_COLUMN, TARGET_COLUMN, TEXT_COLUMN, "split"])
    df[KEY_COLUMN] = df[KEY_COLUMN].astype(str).str.replace("\\", "/", regex=False).str.strip()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip().str.lower()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    if not include_xxx:
        df = df[df[TARGET_COLUMN] != "xxx"].copy()
    return df.drop_duplicates(subset=[KEY_COLUMN]).reset_index(drop=True)


def load_family_frame(repo_root: Path, family: str, selected: bool) -> tuple[pd.DataFrame, list[str]]:
    source_path = repo_root / FAMILY_SOURCES[family]
    feature_cols = family_feature_columns(repo_root, family, selected)
    usecols = [KEY_COLUMN, *feature_cols]
    df = pd.read_csv(source_path, usecols=usecols)
    df[KEY_COLUMN] = df[KEY_COLUMN].astype(str).str.replace("\\", "/", regex=False).str.strip()

    prefixed = {col: f"{family}__{col}" for col in feature_cols}
    df = df.rename(columns=prefixed)
    out_cols = list(prefixed.values())
    df[out_cols] = df[out_cols].replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates(subset=[KEY_COLUMN])
    return df, out_cols


def build_model_frame(
    repo_root: Path,
    base_df: pd.DataFrame,
    spec: ModelSpec,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df = base_df.copy()
    features_by_family: dict[str, list[str]] = {}

    for family_spec in spec.families:
        family_df, prefixed_cols = load_family_frame(
            repo_root,
            family=family_spec.name,
            selected=family_spec.selected,
        )
        df = df.merge(family_df, on=KEY_COLUMN, how="inner")
        features_by_family[family_spec.name] = prefixed_cols

    return df.reset_index(drop=True), features_by_family


def make_text_transformer(name: str) -> BaseEstimator:
    if name == "word_tfidf_low":
        return TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 1),
            max_features=1000,
            min_df=2,
            sublinear_tf=True,
        )
    if name == "word_tfidf_medium":
        return TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=6000,
            min_df=2,
            sublinear_tf=True,
        )
    if name == "word_char_tfidf_high":
        return FeatureUnion(
            transformer_list=[
                (
                    "word",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        analyzer="word",
                        ngram_range=(1, 2),
                        max_features=10000,
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        lowercase=True,
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        max_features=5000,
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
    raise KeyError(f"Unknown text vectorizer: {name}")


def make_classifier(name: str) -> BaseEstimator:
    if name == "logreg_low":
        return LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
    if name == "logreg_medium":
        return LogisticRegression(
            solver="saga",
            class_weight="balanced",
            max_iter=1200,
            C=1.0,
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
    if name == "sgd_high":
        return SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=1e-4,
            l1_ratio=0.10,
            class_weight="balanced",
            max_iter=2000,
            tol=1e-4,
            early_stopping=True,
            n_iter_no_change=8,
            random_state=RANDOM_STATE,
        )
    raise KeyError(f"Unknown classifier: {name}")


def make_pipeline(spec: ModelSpec, numeric_cols: list[str]) -> Pipeline:
    transformers: list[tuple[str, BaseEstimator, str | list[str]]] = []
    if spec.text_vectorizer is not None:
        transformers.append(("text", make_text_transformer(spec.text_vectorizer), TEXT_COLUMN))
    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if not transformers:
        raise ValueError(f"{spec.model_id} has neither text nor numeric inputs.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.30,
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", make_classifier(spec.classifier)),
        ]
    )


def split_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Expected non-empty train/test splits in base feature table.")
    return train_df, test_df


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def top_probabilities(model: Pipeline, x_test: pd.DataFrame, labels: list[str], limit: int = 5) -> list[dict[str, Any]]:
    if not hasattr(model, "predict_proba"):
        return []

    proba = model.predict_proba(x_test.head(limit))
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(proba):
        order = np.argsort(row)[::-1][:3]
        rows.append(
            {
                "row": int(idx),
                "top": [
                    {"label": labels[int(label_idx)], "probability": float(row[int(label_idx)])}
                    for label_idx in order
                ],
            }
        )
    return rows


def finite_float(value: float) -> float | None:
    if math.isfinite(value):
        return float(value)
    return None


def save_package(
    *,
    repo_root: Path,
    out_dir: Path,
    spec: ModelSpec,
    model: Pipeline,
    features_by_family: dict[str, list[str]],
    metrics: dict[str, Any],
    train_rows: int,
    test_rows: int,
    include_xxx: bool,
    source_rows: int,
) -> dict[str, Any]:
    model_dir = out_dir / spec.model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path, compress=3)

    numeric_features = [feature for features in features_by_family.values() for feature in features]
    feature_schema = {
        "model_id": spec.model_id,
        "requires_text": spec.text_vectorizer is not None,
        "text_column": TEXT_COLUMN if spec.text_vectorizer is not None else None,
        "text_vectorizer": spec.text_vectorizer,
        "numeric_feature_count": len(numeric_features),
        "numeric_features": numeric_features,
        "features_by_family": features_by_family,
    }
    write_json(model_dir / "feature_schema.json", feature_schema)

    source_tables = {
        family.name: str((repo_root / FAMILY_SOURCES[family.name]).relative_to(repo_root))
        for family in spec.families
    }
    manifest = {
        "model_id": spec.model_id,
        "modality": spec.modality,
        "tier": spec.tier,
        "description": spec.description,
        "model_file": "model.joblib",
        "feature_schema_file": "feature_schema.json",
        "metrics_file": "metrics.json",
        "classes": [str(c) for c in model.classes_],
        "input_contract": {
            "requires_text": spec.text_vectorizer is not None,
            "requires_numeric_features": bool(numeric_features),
            "numeric_feature_count": len(numeric_features),
            "payload_shape": {
                "text": "string transcript; required when requires_text is true",
                "features": "object mapping feature name to numeric value; required when requires_numeric_features is true",
            },
        },
        "components": {
            "text_vectorizer": spec.text_vectorizer,
            "feature_families": [
                {"name": family.name, "selected_features": family.selected}
                for family in spec.families
            ],
            "classifier": spec.classifier,
        },
        "training": {
            "source_rows_after_base_filter": int(source_rows),
            "train_rows": int(train_rows),
            "test_rows": int(test_rows),
            "include_xxx": bool(include_xxx),
            "random_state": RANDOM_STATE,
            "base_table": str(BASE_TABLE),
            "source_tables": source_tables,
        },
        "metrics_summary": {
            "accuracy": finite_float(metrics["accuracy"]),
            "f1_macro": finite_float(metrics["f1_macro"]),
            "f1_weighted": finite_float(metrics["f1_weighted"]),
        },
    }
    write_json(model_dir / "manifest.json", manifest)
    write_json(model_dir / "metrics.json", metrics)
    return manifest


def train_one(repo_root: Path, out_dir: Path, base_df: pd.DataFrame, spec: ModelSpec, include_xxx: bool) -> dict[str, Any]:
    print(f"\n[{spec.model_id}] building frame")
    frame, features_by_family = build_model_frame(repo_root, base_df, spec)
    train_df, test_df = split_frame(frame)
    numeric_cols = [feature for features in features_by_family.values() for feature in features]
    input_cols = ([TEXT_COLUMN] if spec.text_vectorizer is not None else []) + numeric_cols

    print(
        f"[{spec.model_id}] train={len(train_df)} test={len(test_df)} "
        f"text={spec.text_vectorizer is not None} numeric_features={len(numeric_cols)}"
    )

    model = make_pipeline(spec, numeric_cols)
    model.fit(train_df[input_cols], train_df[TARGET_COLUMN])
    predictions = model.predict(test_df[input_cols])
    labels = [str(c) for c in model.classes_]
    metrics = compute_metrics(test_df[TARGET_COLUMN], predictions, labels)
    metrics["proba_preview"] = top_probabilities(model, test_df[input_cols], labels)

    manifest = save_package(
        repo_root=repo_root,
        out_dir=out_dir,
        spec=spec,
        model=model,
        features_by_family=features_by_family,
        metrics=metrics,
        train_rows=len(train_df),
        test_rows=len(test_df),
        include_xxx=include_xxx,
        source_rows=len(frame),
    )
    print(
        f"[{spec.model_id}] accuracy={metrics['accuracy']:.4f} "
        f"f1_macro={metrics['f1_macro']:.4f} f1_weighted={metrics['f1_weighted']:.4f}"
    )
    return manifest


def export_models(repo_root: Path, out_dir: Path, include_xxx: bool, clean: bool) -> list[dict[str, Any]]:
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_df = load_base_frame(repo_root, include_xxx=include_xxx)
    manifests: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        manifests.append(train_one(repo_root, out_dir, base_df, spec, include_xxx))

    registry = {
        "exported_at": pd.Timestamp.utcnow().isoformat(),
        "model_count": len(manifests),
        "models": [
            {
                "model_id": manifest["model_id"],
                "modality": manifest["modality"],
                "tier": manifest["tier"],
                "path": manifest["model_id"],
                "accuracy": manifest["metrics_summary"]["accuracy"],
                "f1_macro": manifest["metrics_summary"]["f1_macro"],
                "numeric_feature_count": manifest["input_contract"]["numeric_feature_count"],
                "requires_text": manifest["input_contract"]["requires_text"],
            }
            for manifest in manifests
        ],
    }
    write_json(out_dir / "registry.json", registry)
    return manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export nine API-ready SER model packages.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("api_models/artifacts"),
        help="Directory where model packages will be written.",
    )
    parser.add_argument(
        "--exclude-xxx",
        action="store_true",
        help="Drop the IEMOCAP xxx/unknown class before training.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete the output directory before exporting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(Path.cwd().resolve())
    out_dir = (repo_root / args.out_dir).resolve()
    export_models(
        repo_root=repo_root,
        out_dir=out_dir,
        include_xxx=not args.exclude_xxx,
        clean=not args.no_clean,
    )
    print(f"\nSaved model registry: {out_dir / 'registry.json'}")


if __name__ == "__main__":
    main()
