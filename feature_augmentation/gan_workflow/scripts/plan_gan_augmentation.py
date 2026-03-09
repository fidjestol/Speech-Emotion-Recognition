from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

DEFAULT_EXCLUDE_COLS = {
    "path",
    "session",
    "method",
    "gender",
    "emotion",
    "n_annotators",
    "agreement",
    "split",
    "utt_id",
    "text",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan class-balancing targets for GAN feature augmentation."
    )
    parser.add_argument("--train-csv", required=True, help="Training feature CSV path")
    parser.add_argument("--label-col", default="emotion", help="Label column name")
    parser.add_argument(
        "--mode",
        choices=["max", "median", "fixed"],
        default="max",
        help="Target policy for class balancing",
    )
    parser.add_argument(
        "--exclude-cols",
        default=",".join(sorted(DEFAULT_EXCLUDE_COLS)),
        help="Comma-separated columns to exclude from feature matrix",
    )
    parser.add_argument(
        "--fixed-targets-json",
        default="",
        help="JSON file with class->target mapping, required for mode=fixed",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Optional output JSON path for the generated plan",
    )
    return parser.parse_args()


def split_csv_arg(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def compute_targets(
    counts: dict[str, int],
    mode: str,
    fixed_targets: dict[str, int] | None = None,
) -> dict[str, int]:
    if not counts:
        raise ValueError("No labels found in the input.")

    if mode == "max":
        target = max(counts.values())
        return {label: target for label in counts}

    if mode == "median":
        target = int(median(counts.values()))
        return {label: max(count, target) for label, count in counts.items()}

    if mode == "fixed":
        if not fixed_targets:
            raise ValueError("mode=fixed requires --fixed-targets-json")
        out: dict[str, int] = {}
        for label, count in counts.items():
            requested = int(fixed_targets.get(label, count))
            out[label] = max(requested, count)
        return out

    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    args = parse_args()
    train_csv = Path(args.train_csv)
    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")

    header_cols = list(pd.read_csv(train_csv, nrows=0).columns)
    if args.label_col not in header_cols:
        raise ValueError(f"Label column '{args.label_col}' not found in {train_csv}")

    labels = pd.read_csv(train_csv, usecols=[args.label_col])[args.label_col].astype(str).str.strip()
    class_counts = labels.value_counts().to_dict()
    class_counts = {str(k): int(v) for k, v in class_counts.items()}

    exclude_cols = set(split_csv_arg(args.exclude_cols))
    exclude_cols.add(args.label_col)
    feature_candidates = [c for c in header_cols if c not in exclude_cols]

    # Sample rows are enough to infer dtype-like behavior for planning.
    sample_rows = 2048
    sample_df = pd.read_csv(train_csv, usecols=feature_candidates, nrows=sample_rows)
    numeric_features = sample_df.select_dtypes(include=["number"]).columns.tolist()

    fixed_targets: dict[str, int] | None = None
    if args.mode == "fixed":
        fixed_path = Path(args.fixed_targets_json)
        if not fixed_path.exists():
            raise FileNotFoundError(
                f"Fixed targets JSON not found: {fixed_path}. Required for mode=fixed."
            )
        with fixed_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected object in {fixed_path}, got {type(loaded)}")
        fixed_targets = {str(k): int(v) for k, v in loaded.items()}

    target_counts = compute_targets(class_counts, args.mode, fixed_targets=fixed_targets)
    synthetic_needed = {
        label: int(max(0, target_counts[label] - class_counts[label])) for label in class_counts
    }

    plan: dict[str, Any] = {
        "train_csv": str(train_csv),
        "label_col": args.label_col,
        "mode": args.mode,
        "rows_total": int(len(labels)),
        "feature_columns_total": int(len(feature_candidates)),
        "numeric_feature_columns": int(len(numeric_features)),
        "class_counts": class_counts,
        "target_counts": target_counts,
        "synthetic_needed": synthetic_needed,
        "synthetic_total": int(sum(synthetic_needed.values())),
    }

    print(json.dumps(plan, indent=2))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(plan, handle, indent=2)
        print(f"\nSaved plan to: {out_path}")


if __name__ == "__main__":
    main()
