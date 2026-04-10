from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_selection.common import (
    METADATA_CANDIDATES,
    DEFAULT_EXCLUDED_EMOTIONS,
    DEFAULT_REQUIRE_AGREEMENT,
    filter_feature_frame,
    resolve_feature_source,
    variant_name,
)

DEFAULT_FAMILY_KEYS: tuple[str, ...] = (
    "mfcc_raw",
    "prosody_energy",
    "prosody_pitch",
    "ssl_hubert",
    "ssl_wav2vec",
    "bert",
    "tfidf",
    "tonality",
    "rhythm_pauses",
    "representations",
)

DEFAULT_CHECKPOINT_BASENAME = "ga_checkpoint"
JOIN_KEY = "path"
TARGET_COLUMN = "emotion"


def family_keys_from_chromosome(chromosome: Sequence[int], family_keys: Sequence[str]) -> list[str]:
    return [family for gene, family in zip(chromosome, family_keys) if int(gene) == 1]


def chromosome_to_string(chromosome: Sequence[int]) -> str:
    return "".join(str(int(bit)) for bit in chromosome)


def string_to_chromosome(encoded: str) -> list[int]:
    return [int(ch) for ch in encoded]


def selected_family_count(chromosome: Sequence[int]) -> int:
    return sum(int(bit) for bit in chromosome)


def ensure_non_empty_chromosome(chromosome: list[int], rng, min_selected: int = 1) -> list[int]:
    if sum(chromosome) >= min_selected:
        return chromosome
    indices = list(range(len(chromosome)))
    rng.shuffle(indices)
    for idx in indices[:min_selected]:
        chromosome[idx] = 1
    return chromosome


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in METADATA_CANDIDATES]


def load_family_frame(
    repo_root: Path,
    family_key: str,
    *,
    include_xxx: bool,
    require_agreement: bool,
    excluded_emotions: Sequence[str],
    deduplicate_on: str = JOIN_KEY,
) -> pd.DataFrame:
    source_path = resolve_feature_source(repo_root, family_key)
    df = pd.read_csv(source_path)
    df = filter_feature_frame(
        df,
        include_xxx=include_xxx,
        require_agreement=require_agreement,
        excluded_emotions=excluded_emotions,
    )
    if deduplicate_on not in df.columns:
        raise KeyError(f"Required join key '{deduplicate_on}' missing in {source_path}")
    df = df.drop_duplicates(subset=[deduplicate_on]).copy()

    family_features = feature_columns(df)
    rename_map = {column: f"{family_key}__{column}" for column in family_features}
    return df.rename(columns=rename_map)


def merge_selected_families(
    repo_root: Path,
    family_keys: Sequence[str],
    *,
    include_xxx: bool,
    require_agreement: bool = DEFAULT_REQUIRE_AGREEMENT,
    excluded_emotions: Sequence[str] = DEFAULT_EXCLUDED_EMOTIONS,
) -> pd.DataFrame:
    if not family_keys:
        raise ValueError("At least one feature family must be selected.")

    merged: pd.DataFrame | None = None
    metadata_columns: list[str] | None = None
    for family_key in family_keys:
        family_df = load_family_frame(
            repo_root,
            family_key,
            include_xxx=include_xxx,
            require_agreement=require_agreement,
            excluded_emotions=excluded_emotions,
        )
        family_feature_cols = [c for c in family_df.columns if c.startswith(f"{family_key}__")]
        if metadata_columns is None:
            metadata_columns = [c for c in family_df.columns if c in METADATA_CANDIDATES]
            merged = family_df[metadata_columns + family_feature_cols].copy()
        else:
            assert merged is not None
            merged = merged.merge(
                family_df[[JOIN_KEY] + family_feature_cols],
                on=JOIN_KEY,
                how="inner",
                validate="one_to_one",
            )

    assert merged is not None
    if TARGET_COLUMN not in merged.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' missing after merge.")
    merged = merged.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    return merged


def artifacts_dir(base_dir: Path, run_name: str, include_xxx: bool) -> Path:
    return base_dir / run_name / variant_name(include_xxx)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def serializable_history(history: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for row in history:
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            if hasattr(value, "item"):
                value = value.item()
            normalized[key] = value
        serialized.append(normalized)
    return serialized
