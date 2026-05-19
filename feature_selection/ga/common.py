from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_selection.common import variant_name

DEFAULT_CHECKPOINT_BASENAME = "ga_checkpoint"
DEFAULT_ARTIFACTS_ROOT = Path("feature_selection/ga/artifacts")
DEFAULT_GROUP_BLOCK_SIZE = 128
TARGET_COLUMN = "emotion"
JOIN_KEY = "path"


def artifacts_dir(base_dir: Path, run_name: str, include_xxx: bool) -> Path:
    return base_dir / run_name / variant_name(include_xxx)


def chromosome_to_string(chromosome: Sequence[int]) -> str:
    return "".join(str(int(bit)) for bit in chromosome)


def string_to_chromosome(encoded: str) -> list[int]:
    return [int(ch) for ch in encoded]


def ensure_min_selected(chromosome: list[int], rng, min_selected: int = 1) -> list[int]:
    if sum(chromosome) >= min_selected:
        return chromosome
    indices = list(range(len(chromosome)))
    rng.shuffle(indices)
    for idx in indices[:min_selected]:
        chromosome[idx] = 1
    return chromosome


def group_keys_from_chromosome(chromosome: Sequence[int], group_keys: Sequence[str]) -> list[str]:
    return [group for gene, group in zip(chromosome, group_keys) if int(gene) == 1]


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def serializable_history(history: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in history:
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            if hasattr(value, "item"):
                value = value.item()
            normalized[key] = value
        rows.append(normalized)
    return rows

