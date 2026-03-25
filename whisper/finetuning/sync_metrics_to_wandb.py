from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload post-hoc Whisper SER metrics from saved run artifacts to Weights & Biases."
    )
    parser.add_argument("--run-dir", required=True, help="Path to a run directory under whisper/finetuning/artifacts.")
    parser.add_argument("--project", required=True, help="Weights & Biases project name.")
    parser.add_argument("--entity", default=None, help="Optional Weights & Biases entity.")
    parser.add_argument("--name", default=None, help="Optional W&B run name.")
    parser.add_argument("--group", default=None, help="Optional W&B run group.")
    parser.add_argument(
        "--job-type",
        default="posthoc-metrics",
        help="W&B job type for the uploaded summary run.",
    )
    parser.add_argument(
        "--tags",
        default="posthoc,metrics",
        help="Comma-separated W&B tags.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_variant_metrics(run_dir: Path) -> pd.DataFrame:
    weighted_path = run_dir / "variant_comparison.csv"
    unweighted_path = run_dir / "variant_comparison_unweighted.csv"
    if not weighted_path.exists():
        raise FileNotFoundError(f"Missing weighted metrics file: {weighted_path}")
    if not unweighted_path.exists():
        raise FileNotFoundError(f"Missing unweighted metrics file: {unweighted_path}")

    weighted_df = pd.read_csv(weighted_path)
    unweighted_df = pd.read_csv(unweighted_path)

    keep_unweighted = [
        "variant",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_micro",
        "recall_micro",
        "f1_micro",
    ]
    return weighted_df.merge(unweighted_df[keep_unweighted], on="variant", suffixes=("", "_posthoc"))


def flatten_variant_records(metrics_df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for record in metrics_df.to_dict(orient="records"):
        variant = record["variant"]
        for key, value in record.items():
            if key == "variant":
                continue
            summary[f"{variant}/{key}"] = value
    return summary


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    metrics_df = merge_variant_metrics(run_dir)
    run_config_path = run_dir / "run_config.json"
    environment_summary_path = run_dir / "environment_summary.json"

    config: dict[str, Any] = {}
    if run_config_path.exists():
        config["training_run_config"] = load_json(run_config_path)
    if environment_summary_path.exists():
        config["environment_summary"] = load_json(environment_summary_path)

    run_name = args.name or f"{run_dir.name}-posthoc-metrics"
    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]

    wandb_run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        group=args.group or run_dir.name,
        job_type=args.job_type,
        config=config,
        tags=tags,
        reinit="finish_previous",
    )

    try:
        wandb_run.summary["source_run_dir"] = str(run_dir)
        for key, value in flatten_variant_records(metrics_df).items():
            wandb_run.summary[key] = value

        metrics_table = wandb.Table(dataframe=metrics_df)
        wandb.log({"posthoc_variant_metrics": metrics_table})

        artifact = wandb.Artifact(f"{run_dir.name}-posthoc-metrics", type="metrics")
        for rel_path in [
            "variant_comparison.csv",
            "variant_comparison.json",
            "variant_comparison_unweighted.csv",
            "variant_comparison_unweighted.json",
        ]:
            path = run_dir / rel_path
            if path.exists():
                artifact.add_file(path, name=rel_path)
        wandb.log_artifact(artifact)
    finally:
        wandb.finish()

    print(f"Uploaded post-hoc metrics for: {run_dir}")
    print(f"W&B run name: {run_name}")


if __name__ == "__main__":
    main()
