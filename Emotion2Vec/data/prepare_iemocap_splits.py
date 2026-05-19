from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from Emotion2Vec import EMOTION_LABELS


DEFAULT_METADATA_CSV = "datasets/IEMOCAP/iemocap_full_dataset.csv"
DEFAULT_EMBEDDINGS_ROOT = "extracted_features/emotion2vec_base/frame"
DEFAULT_OUTPUT_DIR = "datasets/IEMOCAP/splits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare train/val/test CSV splits for the 6-way downstream emotion2vec classifier "
            "from the repo's IEMOCAP metadata."
        )
    )
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV)
    parser.add_argument("--embeddings-root", default=DEFAULT_EMBEDDINGS_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label-col", default="emotion")
    parser.add_argument("--path-col", default="path")
    parser.add_argument("--agreement-col", default="agreement")
    parser.add_argument("--sample-id-col", default=None)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(EMOTION_LABELS),
        help="Allowed downstream labels. Default is the requested 6-way setup.",
    )
    parser.add_argument(
        "--min-agreement",
        type=int,
        default=1,
        help="Default keeps only rows with annotator agreement.",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--require-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop rows whose derived .pt embedding file does not exist.",
    )
    parser.add_argument(
        "--prefix",
        default="iemocap_emotion2vec_base_6way",
        help="Output file prefix before _train/_val/_test.csv.",
    )
    return parser.parse_args()


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root from the current path.")


def resolve_repo_path(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def normalize_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(Path(__file__).resolve())
    metadata_csv = resolve_repo_path(args.metadata_csv, repo_root)
    embeddings_root = resolve_repo_path(args.embeddings_root, repo_root)
    output_dir = resolve_repo_path(args.output_dir, repo_root)

    frame = pd.read_csv(metadata_csv)
    frame[args.label_col] = normalize_series(frame[args.label_col])
    allowed_labels = [label.strip().lower() for label in args.labels]
    frame = frame[frame[args.label_col].isin(allowed_labels)].copy()
    frame = frame[frame[args.agreement_col].fillna(0).astype(int) >= args.min_agreement].copy()

    if frame.empty:
        raise RuntimeError("No IEMOCAP rows remained after label/agreement filtering.")

    frame["label"] = frame[args.label_col]
    frame["embedding_path"] = frame[args.path_col].astype(str).str.replace("\\", "/", regex=False).map(
        lambda rel: str((embeddings_root / Path(rel).with_suffix(".pt")).resolve())
    )
    frame["embedding_exists"] = frame["embedding_path"].map(lambda path: Path(path).exists())

    if args.require_embeddings:
        frame = frame[frame["embedding_exists"]].copy()

    if args.sample_id_col and args.sample_id_col in frame.columns:
        frame["sample_id"] = frame[args.sample_id_col].astype(str).str.strip()
    else:
        frame["sample_id"] = frame[args.path_col].astype(str).map(lambda rel: Path(rel).stem)

    if frame.empty:
        raise RuntimeError("No rows remained after enforcing existing embeddings.")

    train_val_df, test_df = train_test_split(
        frame,
        test_size=args.test_size,
        stratify=frame["label"],
        random_state=args.random_state,
    )
    val_relative = args.val_size / (1.0 - args.test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative,
        stratify=train_val_df["label"],
        random_state=args.random_state,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"{args.prefix}_train.csv"
    val_path = output_dir / f"{args.prefix}_val.csv"
    test_path = output_dir / f"{args.prefix}_test.csv"
    summary_path = output_dir / f"{args.prefix}_summary.json"

    columns = [
        "sample_id",
        "label",
        "embedding_path",
        args.label_col,
        args.path_col,
        args.agreement_col,
        "embedding_exists",
    ]
    passthrough_columns = [column for column in ("session", "method", "gender") if column in frame.columns]
    columns.extend(passthrough_columns)

    train_df.loc[:, columns].to_csv(train_path, index=False)
    val_df.loc[:, columns].to_csv(val_path, index=False)
    test_df.loc[:, columns].to_csv(test_path, index=False)

    def label_counts(split_df: pd.DataFrame) -> dict[str, int]:
        counts = split_df["label"].value_counts().sort_index()
        return {str(key): int(value) for key, value in counts.items()}

    summary: dict[str, Any] = {
        "metadata_csv": str(metadata_csv),
        "embeddings_root": str(embeddings_root),
        "output_dir": str(output_dir),
        "allowed_labels": allowed_labels,
        "min_agreement": args.min_agreement,
        "require_embeddings": args.require_embeddings,
        "random_state": args.random_state,
        "sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "label_counts": {
            "train": label_counts(train_df),
            "val": label_counts(val_df),
            "test": label_counts(test_df),
        },
        "paths": {
            "train_csv": str(train_path),
            "val_csv": str(val_path),
            "test_csv": str(test_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Train CSV: {train_path}")
    print(f"Val CSV:   {val_path}")
    print(f"Test CSV:  {test_path}")
    print(f"Summary:   {summary_path}")


if __name__ == "__main__":
    main()

