from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_MACHINE_NAME = "macbook"
DEFAULT_EXCLUDED_EMOTIONS = ("sur", "fea", "oth", "dis")
DEFAULT_REQUIRE_AGREEMENT = True
DEFAULT_INCLUDE_XXX = True
DEFAULT_USE_VARIANT_ARTIFACT_DIRS = True

METADATA_CANDIDATES = {
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

FEATURE_SOURCE_RELATIVE_PATHS: dict[str, Path] = {
    "mfcc_raw": Path("extracted_features/mfcc/mfcc_features.csv"),
    "mfcc_normalized": Path("extracted_features/mfcc/mfcc_features_normalized.csv"),
    "prosody_energy": Path("extracted_features/prosody_energy/prosody_energy_features.csv"),
    "prosody_pitch": Path("extracted_features/prosody_pitch/prosody_pitch_features.csv"),
    "bert": Path("extracted_features/text/bert_embeddings.csv"),
    "tfidf": Path("extracted_features/text/tfidf_features.csv"),
    "tfidf_kernel_pca": Path("extracted_features/text/tfidf_features.csv"),
    "representations": Path("extracted_features/representations/representations_features.csv"),
    "rhythm_pauses": Path("extracted_features/rhythm_pauses/rhythm_pauses_features.csv"),
    "tonality": Path("extracted_features/tonality/tonality_features.csv"),
    "ssl_embeddings": Path("extracted_features/ssl_embeddings/ssl_embeddings_features.csv"),
    "ssl_hubert": Path("extracted_features/ssl_embeddings/ssl_embeddings_hubert_features.csv"),
    "ssl_wav2vec": Path("extracted_features/ssl_embeddings/ssl_embeddings_wav2vec_features.csv"),
    "combined": Path("extracted_features/combined/all_families_features.csv"),
}

PCA_CPU_SKIP_NOTEBOOKS = {
    "kernel_pca_tfidf.ipynb",
    "pca_artifact_xxx_analysis.ipynb",
    "pca_tfidf_gpu.ipynb",
}


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists():
            return path
    raise FileNotFoundError("Could not locate repository root (missing pyproject.toml).")


def parse_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def machine_name_from_env(default: str = DEFAULT_MACHINE_NAME) -> str:
    return str(os.getenv("SER_MACHINE", default)).strip().lower() or default


def include_xxx_from_env(default: bool = DEFAULT_INCLUDE_XXX) -> bool:
    return parse_bool(os.getenv("SER_INCLUDE_XXX"), default=default)


def run_both_xxx_variants_from_env(default: bool = True) -> bool:
    return parse_bool(os.getenv("SER_RUN_BOTH_XXX_VARIANTS"), default=default)


def use_variant_artifact_dirs_from_env(default: bool = DEFAULT_USE_VARIANT_ARTIFACT_DIRS) -> bool:
    return parse_bool(os.getenv("SER_USE_VARIANT_ARTIFACT_DIRS"), default=default)


def variant_name(include_xxx: bool) -> str:
    return "with_xxx" if include_xxx else "without_xxx"


def variants_to_run(default_both: bool = True, default_include_xxx: bool = DEFAULT_INCLUDE_XXX) -> list[bool]:
    if run_both_xxx_variants_from_env(default=default_both):
        return [True, False]
    return [include_xxx_from_env(default=default_include_xxx)]


def resolve_feature_source(repo_root: Path, dataset_key: str) -> Path:
    if dataset_key not in FEATURE_SOURCE_RELATIVE_PATHS:
        raise KeyError(f"Unknown dataset key: {dataset_key}")
    return repo_root / FEATURE_SOURCE_RELATIVE_PATHS[dataset_key]


def resolve_variant_artifact_dir(
    base_dir: Path,
    *,
    include_xxx: bool,
    use_variant_dirs: bool = DEFAULT_USE_VARIANT_ARTIFACT_DIRS,
) -> Path:
    if not use_variant_dirs:
        return base_dir
    return base_dir.parent / f"{base_dir.name}_{variant_name(include_xxx)}"


def resolve_pca_artifact_dir(
    repo_root: Path,
    *,
    include_xxx: bool,
    use_variant_dirs: bool = DEFAULT_USE_VARIANT_ARTIFACT_DIRS,
) -> Path:
    base_dir = repo_root / "feature_selection" / "pca" / "select_pca" / "artifacts"
    return resolve_variant_artifact_dir(
        base_dir,
        include_xxx=include_xxx,
        use_variant_dirs=use_variant_dirs,
    )


def resolve_anova_artifact_dir(
    repo_root: Path,
    *,
    include_xxx: bool,
    use_variant_dirs: bool = DEFAULT_USE_VARIANT_ARTIFACT_DIRS,
) -> Path:
    base_dir = repo_root / "feature_selection" / "anova" / "select_k_best_anova" / "artifacts"
    return resolve_variant_artifact_dir(
        base_dir,
        include_xxx=include_xxx,
        use_variant_dirs=use_variant_dirs,
    )


def resolve_cpu_parallel_config(machine_name: str | None = None) -> tuple[int, str]:
    machine = (machine_name or machine_name_from_env()).strip().lower()
    if machine == "macbook":
        return 1, "sequential"
    return -1, "auto"


def resolve_random_forest_jobs(machine_name: str | None = None) -> int:
    machine = (machine_name or machine_name_from_env()).strip().lower()
    if machine == "macbook":
        return 1
    return -1


def resolve_pca_component_count(
    requested_components: int | float | None,
    *,
    n_samples: int,
    n_features: int,
) -> int | float:
    max_components = max(1, min(int(n_samples), int(n_features)))
    if requested_components is None:
        return max_components
    if isinstance(requested_components, float):
        if 0 < requested_components < 1:
            return requested_components
        return max(1, min(int(round(requested_components)), max_components))
    return max(1, min(int(requested_components), max_components))


def filter_feature_frame(
    df: "pd.DataFrame",
    *,
    target_col: str = "emotion",
    include_xxx: bool = DEFAULT_INCLUDE_XXX,
    require_agreement: bool = DEFAULT_REQUIRE_AGREEMENT,
    excluded_emotions: Iterable[str] = DEFAULT_EXCLUDED_EMOTIONS,
) -> "pd.DataFrame":
    import pandas as pd

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in frame.")

    work_df = df.copy()
    work_df[target_col] = work_df[target_col].astype(str).str.strip().str.lower()

    if excluded_emotions:
        excluded = {str(label).strip().lower() for label in excluded_emotions}
        work_df = work_df[~work_df[target_col].isin(excluded)].copy()

    if not include_xxx:
        work_df = work_df[work_df[target_col] != "xxx"].copy()

    if require_agreement and "agreement" in work_df.columns:
        agreement = pd.to_numeric(work_df["agreement"], errors="coerce")
        if include_xxx:
            work_df = work_df[(work_df[target_col] == "xxx") | (agreement > 0)].copy()
        else:
            work_df = work_df[agreement > 0].copy()

    return work_df.reset_index(drop=True)


def pca_runner_skip_notebooks(machine_name: str | None = None) -> set[str]:
    machine = (machine_name or machine_name_from_env()).strip().lower()
    if machine == "macbook":
        return set(PCA_CPU_SKIP_NOTEBOOKS)
    return {"pca_artifact_xxx_analysis.ipynb"}
