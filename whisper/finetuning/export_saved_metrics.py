from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute additional summary metrics from saved Whisper SER prediction tables."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a run directory under whisper/finetuning/artifacts.",
    )
    return parser.parse_args()


def compute_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    y_true = predictions["true_label"]
    y_pred = predictions["pred_label"]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def variant_dirs(run_dir: Path) -> list[Path]:
    return sorted(
        path for path in run_dir.iterdir() if path.is_dir() and (path / "reports" / "test_predictions.csv").exists()
    )


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    summaries: list[dict[str, object]] = []
    for variant_dir in variant_dirs(run_dir):
        predictions_path = variant_dir / "reports" / "test_predictions.csv"
        predictions = pd.read_csv(predictions_path)
        metrics = compute_metrics(predictions)
        record: dict[str, object] = {"variant": variant_dir.name, **metrics}
        summaries.append(record)

        metrics_path_csv = variant_dir / "reports" / "test_metrics_unweighted.csv"
        metrics_path_json = variant_dir / "reports" / "test_metrics_unweighted.json"
        pd.DataFrame([record]).to_csv(metrics_path_csv, index=False)
        save_json(metrics_path_json, record)

    if not summaries:
        raise RuntimeError(f"No variant prediction tables found under: {run_dir}")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(run_dir / "variant_comparison_unweighted.csv", index=False)
    save_json(run_dir / "variant_comparison_unweighted.json", summaries)

    print(summary_df.to_string(index=False))
    print(f"Wrote: {run_dir / 'variant_comparison_unweighted.csv'}")


if __name__ == "__main__":
    main()
