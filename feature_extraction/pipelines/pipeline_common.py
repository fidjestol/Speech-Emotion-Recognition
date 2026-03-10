from __future__ import annotations

import os
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

KEY_COLUMN = "path"
METADATA_COLUMNS = [
    "path",
    "session",
    "method",
    "gender",
    "emotion",
    "n_annotators",
    "agreement",
]

FAMILY_SOURCES = {
    "mfcc_raw": Path("extracted_features/mfcc/mfcc_features.csv"),
    "mfcc_normalized": Path("extracted_features/mfcc/mfcc_features_normalized.csv"),
    "prosody_energy": Path("extracted_features/prosody_energy/prosody_energy_features.csv"),
    "prosody_pitch": Path("extracted_features/prosody_pitch/prosody_pitch_features.csv"),
    "tfidf": Path("extracted_features/text/tfidf_features.csv"),
    "bert": Path("extracted_features/text/bert_embeddings.csv"),
    "representations": Path("extracted_features/representations/representations_features.csv"),
    "rhythm_pauses": Path("extracted_features/rhythm_pauses/rhythm_pauses_features.csv"),
    "ssl_embeddings": Path("extracted_features/ssl_embeddings/ssl_embeddings_features.csv"),
    "tonality": Path("extracted_features/tonality/tonality_features.csv"),
}

# Backward/forward-compatible source aliases for families that have renamed outputs.
FAMILY_SOURCE_ALIASES: dict[str, tuple[Path, ...]] = {
    "ssl_embeddings": (
        Path("extracted_features/ssl_embeddings/ssl_embeddings_wav2vec_features.csv"),
        Path("extracted_features/ssl_embeddings/ssl_embeddings_hubert_features.csv"),
    ),
}

FAMILY_GENERATOR_NOTEBOOKS = {
    "mfcc_raw": Path("feature_extraction/feature_families/cepstral_mfcc.ipynb"),
    "mfcc_normalized": Path("feature_extraction/feature_families/cepstral_mfcc.ipynb"),
    "prosody_energy": Path("feature_extraction/feature_families/prosody_energy.ipynb"),
    "prosody_pitch": Path("feature_extraction/feature_families/prosody_pitch.ipynb"),
    "tfidf": Path("feature_extraction/feature_families/text_tfidf.ipynb"),
    "bert": Path("feature_extraction/feature_families/text_bert_embeddings.ipynb"),
    "representations": Path("feature_extraction/feature_families/representations.ipynb"),
    "rhythm_pauses": Path("feature_extraction/feature_families/rhythm_pauses.ipynb"),
    "ssl_embeddings": Path("feature_extraction/feature_families/ssl_embeddings_wav2vec.ipynb"),
    "tonality": Path("feature_extraction/feature_families/tonality.ipynb"),
}

DEFAULT_ALL_FAMILIES = list(FAMILY_SOURCES)
ANOVA_FAMILIES = [
    "mfcc_raw",
    "mfcc_normalized",
    "prosody_energy",
    "prosody_pitch",
    "tfidf",
    "bert",
]


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists():
            return path
    raise FileNotFoundError("Could not locate repository root (missing pyproject.toml).")


def normalize_path_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("\\", "/", regex=False).str.strip()


def prefixed_name(family: str, feature_name: str) -> str:
    return f"{family}__{feature_name}"


def _resolve_family_csv_path(repo_root: Path, family: str) -> Path:
    """Return the first existing CSV path for a family (primary or alias)."""

    primary = repo_root / FAMILY_SOURCES[family]
    if primary.exists():
        return primary

    for alias_rel in FAMILY_SOURCE_ALIASES.get(family, ()):
        alias = repo_root / alias_rel
        if alias.exists():
            return alias

    return primary


@contextmanager
def _pushd(target_dir: Path):
    prev = Path.cwd()
    os.chdir(target_dir)
    try:
        yield
    finally:
        os.chdir(prev)


def load_base_metadata(
    repo_root: Path,
    include_xxx: bool = True,
    require_agreement: bool = True,
    excluded_emotions: tuple[str, ...] = ("sur", "fea", "oth", "dis"),
) -> pd.DataFrame:
    meta_csv = repo_root / "datasets" / "IEMOCAP" / "iemocap_full_dataset.csv"
    if not meta_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {meta_csv}")

    df = pd.read_csv(meta_csv)
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
    df["method"] = df["method"].astype(str).str.strip().str.lower()
    df["gender"] = df["gender"].astype(str).str.strip().str.upper()
    df[KEY_COLUMN] = normalize_path_series(df[KEY_COLUMN])

    if excluded_emotions:
        excluded = {str(label).strip().lower() for label in excluded_emotions}
        df = df[~df["emotion"].isin(excluded)].copy()

    if not include_xxx:
        df = df[df["emotion"] != "xxx"].copy()
    if require_agreement:
        # Keep unlabeled rows when xxx is included; enforce agreement on labeled rows.
        if include_xxx:
            df = df[(df["emotion"] == "xxx") | (df["agreement"] > 0)].copy()
        else:
            df = df[df["agreement"] > 0].copy()

    return df.reset_index(drop=True)


def load_family_frame(
    repo_root: Path,
    family: str,
    prefix_features: bool = True,
    include_features: list[str] | None = None,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if family not in FAMILY_SOURCES:
        raise KeyError(f"Unknown family: {family}")

    csv_path = _resolve_family_csv_path(repo_root, family)
    info: dict[str, Any] = {
        "family": family,
        "csv_path": str(csv_path),
        "available": False,
        "rows": 0,
        "features": 0,
        "status": "missing_file",
    }
    if not csv_path.exists():
        return None, info

    cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if KEY_COLUMN not in cols:
        raise ValueError(f"Family '{family}' is missing key column '{KEY_COLUMN}'")

    all_feature_cols = [c for c in cols if c not in METADATA_COLUMNS]
    feature_cols = all_feature_cols
    missing_requested: list[str] = []
    if include_features is not None:
        feature_cols = [c for c in include_features if c in all_feature_cols]
        missing_requested = [c for c in include_features if c not in all_feature_cols]

    if not feature_cols:
        info["status"] = "no_features"
        if include_features is not None:
            info["status"] = "no_requested_features"
            info["requested_features"] = int(len(include_features))
            info["missing_requested"] = int(len(missing_requested))
        return None, info

    selected_cols = [KEY_COLUMN, *feature_cols]
    df = pd.read_csv(csv_path, usecols=selected_cols)
    df[KEY_COLUMN] = normalize_path_series(df[KEY_COLUMN])
    out_df = df[selected_cols].copy()
    if prefix_features:
        out_df = out_df.rename(columns={c: prefixed_name(family, c) for c in feature_cols})

    out_df = out_df.drop_duplicates(subset=[KEY_COLUMN], keep="first")

    info.update(
        {
            "available": True,
            "rows": int(len(out_df)),
            "features": int(len(feature_cols)),
            "status": "ok",
            "requested_features": int(len(include_features)) if include_features is not None else None,
            "missing_requested": int(len(missing_requested)) if include_features is not None else None,
        }
    )
    return out_df, info


def load_family_frames(
    repo_root: Path,
    families: list[str],
    prefix_features: bool = True,
    include_features_by_family: dict[str, list[str]] | None = None,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    frames: dict[str, pd.DataFrame] = {}
    report: list[dict[str, Any]] = []

    for family in families:
        frame, info = load_family_frame(
            repo_root,
            family=family,
            prefix_features=prefix_features,
            include_features=(include_features_by_family or {}).get(family),
        )
        report.append(info)
        if frame is not None:
            frames[family] = frame

    return frames, report


def missing_family_sources(repo_root: Path, families: list[str]) -> list[str]:
    missing: list[str] = []
    for family in families:
        csv_path = _resolve_family_csv_path(repo_root, family)
        if not csv_path.exists():
            missing.append(family)
    return missing


def run_missing_family_generators(
    repo_root: Path,
    families: list[str],
    execute_timeout: int | None = None,
) -> list[dict[str, Any]]:
    missing = missing_family_sources(repo_root, families)
    if not missing:
        return []

    execution_plan: list[tuple[str, Path]] = []
    seen_notebooks: set[Path] = set()
    for family in families:
        if family not in missing:
            continue
        notebook_rel = FAMILY_GENERATOR_NOTEBOOKS.get(family)
        if notebook_rel is None:
            continue
        notebook_path = repo_root / notebook_rel
        if notebook_path not in seen_notebooks:
            execution_plan.append((family, notebook_path))
            seen_notebooks.add(notebook_path)

    results: list[dict[str, Any]] = []
    for trigger_family, notebook_path in execution_plan:
        row: dict[str, Any] = {
            "trigger_family": trigger_family,
            "notebook": str(notebook_path),
            "status": "pending",
            "error": None,
        }
        if not notebook_path.exists():
            row["status"] = "missing_notebook"
            results.append(row)
            continue

        try:
            import nbformat
            from nbclient import NotebookClient

            nb = nbformat.read(notebook_path, as_version=4)
            with _pushd(notebook_path.parent):
                client = NotebookClient(
                    nb,
                    timeout=execute_timeout,
                    kernel_name="python3",
                )
                client.execute()
            row["status"] = "executed"
        except Exception as exc:  # noqa: BLE001
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
        results.append(row)

    return results


def merge_feature_families(
    base_df: pd.DataFrame,
    family_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    merged = base_df.copy()
    merged[KEY_COLUMN] = normalize_path_series(merged[KEY_COLUMN])

    for _, frame in family_frames.items():
        merged = merged.merge(frame, on=KEY_COLUMN, how="left")

    return merged


def load_selected_feature_lists(
    repo_root: Path,
    artifacts_rel_dir: Path = Path("feature_selection/filter/select_k_best_anova/artifacts"),
) -> dict[str, list[str]]:
    artifacts_dir = repo_root / artifacts_rel_dir
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts dir not found: {artifacts_dir}")

    selected: dict[str, list[str]] = {}
    for json_path in sorted(artifacts_dir.glob("*_selected_features.json")):
        family = json_path.name.replace("_selected_features.json", "")
        selected[family] = json.loads(json_path.read_text(encoding="utf-8"))

    return selected


def select_prefixed_columns(
    selected_by_family: dict[str, list[str]],
    available_columns: list[str],
    families: list[str] | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    available = set(available_columns)
    target_families = families or sorted(selected_by_family)

    selected_cols: list[str] = []
    missing_by_family: dict[str, list[str]] = {}

    for family in target_families:
        requested = selected_by_family.get(family, [])
        missing: list[str] = []

        for feature_name in requested:
            col = prefixed_name(family, feature_name)
            if col in available:
                selected_cols.append(col)
            else:
                missing.append(col)

        if missing:
            missing_by_family[family] = missing

    return selected_cols, missing_by_family


def feature_columns_from_families(df: pd.DataFrame, families: list[str]) -> list[str]:
    prefixes = tuple(f"{name}__" for name in families)
    return [col for col in df.columns if col.startswith(prefixes)]
