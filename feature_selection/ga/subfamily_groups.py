from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_family_selection.genetic_algorithm.common import feature_columns, load_family_frame
from feature_selection.common import DEFAULT_EXCLUDED_EMOTIONS, DEFAULT_REQUIRE_AGREEMENT

from feature_selection.ga.common import DEFAULT_GROUP_BLOCK_SIZE


@dataclass(frozen=True, slots=True)
class SubfamilyGroup:
    key: str
    family: str
    name: str
    columns: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.columns)


def _prefixed(family_key: str, column: str) -> str:
    return f"{family_key}__{column}"


def _numeric_suffix(column: str) -> int | None:
    match = re.search(r"_(\d+)$", column)
    if not match:
        return None
    return int(match.group(1))


def _block_name(prefix: str, index: int, block_size: int) -> str:
    start = (index // block_size) * block_size
    end = start + block_size - 1
    return f"{prefix}_block_{start:04d}_{end:04d}"


def _add_named_group(groups: dict[str, list[str]], name: str, columns: Sequence[str]) -> None:
    clean_columns = [column for column in columns if column]
    if clean_columns:
        groups[name].extend(clean_columns)


def group_feature_columns(
    family_key: str,
    columns: Sequence[str],
    *,
    embedding_block_size: int = DEFAULT_GROUP_BLOCK_SIZE,
) -> list[SubfamilyGroup]:
    """Map raw feature columns for one family into interpretable GA groups."""
    groups: dict[str, list[str]] = defaultdict(list)
    raw_columns = [column.split("__", 1)[1] if "__" in column else column for column in columns]

    for raw, original in zip(raw_columns, columns):
        if raw == "duration_s":
            groups["duration"].append(original)
            continue

        if family_key in {"ssl_hubert", "ssl_wav2vec"}:
            suffix = _numeric_suffix(raw)
            if raw in {"ssl_frames", "ssl_dim"}:
                groups["ssl_metadata"].append(original)
            elif raw.startswith("ssl_mean_") and suffix is not None:
                groups[_block_name("ssl_mean", suffix, embedding_block_size)].append(original)
            elif raw.startswith("ssl_std_") and suffix is not None:
                groups[_block_name("ssl_std", suffix, embedding_block_size)].append(original)
            else:
                groups["ssl_other"].append(original)
            continue

        if family_key == "bert":
            suffix = _numeric_suffix(raw)
            if suffix is not None:
                groups[_block_name("bert_emb", suffix, embedding_block_size)].append(original)
            else:
                groups["bert_other"].append(original)
            continue

        if family_key == "tfidf":
            suffix = _numeric_suffix(raw)
            if suffix is not None:
                groups[_block_name("tfidf", suffix, embedding_block_size)].append(original)
            else:
                groups["tfidf_other"].append(original)
            continue

        if family_key == "mfcc_raw":
            if "delta2" in raw or "ddelta" in raw:
                groups["mfcc_delta2"].append(original)
            elif "delta" in raw:
                groups["mfcc_delta"].append(original)
            elif re.search(r"_c\d{2}_", raw):
                groups["mfcc_coefficients"].append(original)
            elif "mfcc" in raw:
                groups["mfcc_global"].append(original)
            else:
                groups["mfcc_other"].append(original)
            continue

        if family_key == "prosody_energy":
            if "rms" in raw:
                groups["energy_rms"].append(original)
            elif "voiced_ratio" in raw:
                groups["energy_voiced_ratio"].append(original)
            elif "frames" in raw:
                groups["energy_frames"].append(original)
            else:
                groups["energy_other"].append(original)
            continue

        if family_key == "prosody_pitch":
            if "f0_hz" in raw:
                groups["pitch_f0"].append(original)
            elif "voicing_prob" in raw:
                groups["pitch_voicing_probability"].append(original)
            elif "voiced_ratio" in raw:
                groups["pitch_voiced_ratio"].append(original)
            elif "frames" in raw:
                groups["pitch_frames"].append(original)
            else:
                groups["pitch_other"].append(original)
            continue

        if family_key == "rhythm_pauses":
            if "pause" in raw:
                groups["rhythm_pauses"].append(original)
            elif "speech" in raw:
                groups["rhythm_speech"].append(original)
            elif "onset" in raw:
                groups["rhythm_onsets"].append(original)
            else:
                groups["rhythm_other"].append(original)
            continue

        if family_key == "tonality":
            if raw.startswith("chroma_"):
                groups["tonality_chroma"].append(original)
            elif raw.startswith("tonnetz_"):
                groups["tonality_tonnetz"].append(original)
            elif raw.startswith("harmonic_"):
                groups["tonality_harmonic"].append(original)
            else:
                groups["tonality_other"].append(original)
            continue

        if family_key == "representations":
            if raw.startswith("mel_band"):
                suffix = re.search(r"mel_band(\d+)_", raw)
                if suffix:
                    band = int(suffix.group(1))
                    groups[_block_name("log_mel_band", band, 16)].append(original)
                else:
                    groups["log_mel_bands"].append(original)
            elif raw.startswith("log_mel_"):
                groups["log_mel_global"].append(original)
            else:
                groups["representations_other"].append(original)
            continue

        groups["other"].append(original)

    result = [
        SubfamilyGroup(
            key=f"{family_key}::{name}",
            family=family_key,
            name=name,
            columns=tuple(sorted(set(group_columns))),
        )
        for name, group_columns in sorted(groups.items())
        if group_columns
    ]
    return result


def discover_subfamily_groups(
    repo_root: Path,
    family_keys: Sequence[str],
    *,
    include_xxx: bool = False,
    require_agreement: bool = DEFAULT_REQUIRE_AGREEMENT,
    excluded_emotions: Sequence[str] = DEFAULT_EXCLUDED_EMOTIONS,
    embedding_block_size: int = DEFAULT_GROUP_BLOCK_SIZE,
) -> list[SubfamilyGroup]:
    groups: list[SubfamilyGroup] = []
    for family_key in family_keys:
        family_df = load_family_frame(
            repo_root,
            family_key,
            include_xxx=include_xxx,
            require_agreement=require_agreement,
            excluded_emotions=excluded_emotions,
        )
        family_feature_cols = [column for column in feature_columns(family_df) if column.startswith(f"{family_key}__")]
        groups.extend(
            group_feature_columns(
                family_key,
                family_feature_cols,
                embedding_block_size=embedding_block_size,
            )
        )
    return groups


def groups_to_frame(groups: Sequence[SubfamilyGroup]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "group_key": group.key,
                "family": group.family,
                "subfamily": group.name,
                "num_columns": group.size,
                "columns": ",".join(group.columns),
            }
            for group in groups
        ]
    )

