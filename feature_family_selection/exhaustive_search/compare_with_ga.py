from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import wandb

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_selection.common import variant_name
from feature_family_selection.genetic_algorithm.common import save_json
from utils.wandb_multi import init_multi_wandb_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GA and exhaustive feature-family search runs.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--ga-run-name", type=str, required=True)
    parser.add_argument("--exhaustive-run-name", type=str, required=True)
    parser.add_argument(
        "--ga-artifacts-root",
        type=Path,
        default=Path("feature_family_selection/genetic_algorithm/artifacts"),
    )
    parser.add_argument(
        "--exhaustive-artifacts-root",
        type=Path,
        default=Path("feature_family_selection/exhaustive_search/artifacts"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("feature_family_selection/exhaustive_search/comparisons"),
    )
    parser.add_argument(
        "--include-xxx",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compare runs for the chosen variant. Default excludes xxx.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--report-to", choices=["wandb", "none"], default="wandb")
    parser.add_argument("--wandb-project", type=str, default="ser-feature-family-selection")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="compare_ga_vs_exhaustive")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_selected_families(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [item for item in value.split(",") if item]
    raise TypeError(f"Unsupported selected_families payload: {type(value)!r}")


def artifact_dir(base_dir: Path, run_name: str, include_xxx: bool) -> Path:
    return base_dir / run_name / variant_name(include_xxx)


def init_wandb(args: argparse.Namespace, output_dir: Path):
    if args.report_to != "wandb":
        return None
    return init_multi_wandb_run(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{args.run_name}_{variant_name(args.include_xxx)}",
        dir=str(output_dir),
        config={
            **vars(args),
            "repo_root": str(args.repo_root),
            "ga_artifacts_root": str(args.ga_artifacts_root),
            "exhaustive_artifacts_root": str(args.exhaustive_artifacts_root),
            "output_root": str(args.output_root),
            "variant": variant_name(args.include_xxx),
        },
    )


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    ga_dir = artifact_dir(args.ga_artifacts_root, args.ga_run_name, args.include_xxx)
    exhaustive_dir = artifact_dir(args.exhaustive_artifacts_root, args.exhaustive_run_name, args.include_xxx)
    output_dir = artifact_dir(args.output_root, f"{args.ga_run_name}__vs__{args.exhaustive_run_name}", args.include_xxx)
    output_dir.mkdir(parents=True, exist_ok=True)
    run = init_wandb(args, output_dir)
    start_time = time.perf_counter()

    ga_best = load_json(ga_dir / "best_solution.json")
    ga_config = load_json(ga_dir / "run_config.json")
    exhaustive_best = load_json(exhaustive_dir / "best_solution.json")
    exhaustive_config = load_json(exhaustive_dir / "run_config.json")
    exhaustive_results = pd.read_csv(exhaustive_dir / "all_results.csv")

    ga_selected = normalize_selected_families(ga_best["selected_families"])
    exhaustive_selected = normalize_selected_families(exhaustive_best["selected_families"])
    ga_family_set = set(ga_selected)
    exhaustive_family_set = set(exhaustive_selected)

    exhaustive_top_k = exhaustive_results.head(args.top_k).copy()
    ga_rank_matches = exhaustive_results["selected_families_text"] == ",".join(ga_selected)
    ga_rank = int(ga_rank_matches.idxmax()) + 1 if bool(ga_rank_matches.any()) else None

    comparison_row = {
        "variant": variant_name(args.include_xxx),
        "ga_run_name": args.ga_run_name,
        "exhaustive_run_name": args.exhaustive_run_name,
        "ga_best_fitness": float(ga_best["fitness"]),
        "ga_best_macro_f1": float(ga_best["f1_macro"]),
        "ga_best_accuracy": float(ga_best["accuracy"]),
        "ga_selected_families": ga_selected,
        "ga_subset_size": len(ga_selected),
        "exhaustive_best_fitness": float(exhaustive_best["fitness"]),
        "exhaustive_best_macro_f1": float(exhaustive_best["f1_macro"]),
        "exhaustive_best_accuracy": float(exhaustive_best["accuracy"]),
        "exhaustive_selected_families": exhaustive_selected,
        "exhaustive_subset_size": len(exhaustive_selected),
        "fitness_gap_to_optimum": float(exhaustive_best["fitness"]) - float(ga_best["fitness"]),
        "macro_f1_gap_to_optimum": float(exhaustive_best["f1_macro"]) - float(ga_best["f1_macro"]),
        "accuracy_gap_to_optimum": float(exhaustive_best["accuracy"]) - float(ga_best["accuracy"]),
        "exact_family_match": ga_family_set == exhaustive_family_set,
        "ga_rank_in_exhaustive": ga_rank,
        "ga_found_optimum": ga_rank == 1,
        "families_only_in_ga": sorted(ga_family_set - exhaustive_family_set),
        "families_only_in_exhaustive": sorted(exhaustive_family_set - ga_family_set),
        "ga_evaluator_model": ga_config.get("evaluator_model"),
        "exhaustive_evaluator_model": exhaustive_config.get("evaluator_model"),
    }
    comparison_df = pd.DataFrame([comparison_row])

    exhaustive_top_k["rank"] = range(1, len(exhaustive_top_k) + 1)
    exhaustive_top_k = exhaustive_top_k[
        [
            "rank",
            "fitness",
            "f1_macro",
            "accuracy",
            "subset_size",
            "selected_families_text",
            "num_features",
        ]
    ]

    save_json(output_dir / "comparison_summary.json", comparison_row)
    comparison_df.to_csv(output_dir / "comparison_summary.csv", index=False)
    exhaustive_top_k.to_csv(output_dir / "exhaustive_top_k.csv", index=False)

    if run is not None:
        run.log(
            {
                "comparison_table": wandb.Table(dataframe=comparison_df),
                "exhaustive_top_k": wandb.Table(dataframe=exhaustive_top_k),
                "compare/fitness_gap_to_optimum": comparison_row["fitness_gap_to_optimum"],
                "compare/macro_f1_gap_to_optimum": comparison_row["macro_f1_gap_to_optimum"],
                "compare/accuracy_gap_to_optimum": comparison_row["accuracy_gap_to_optimum"],
                "compare/exact_family_match": int(comparison_row["exact_family_match"]),
                "compare/ga_rank_in_exhaustive": comparison_row["ga_rank_in_exhaustive"] or 0,
                "runtime/total_elapsed_seconds": time.perf_counter() - start_time,
            }
        )
        run.summary["ga_found_optimum"] = bool(comparison_row["ga_found_optimum"])
        run.summary["ga_selected_families"] = ga_selected
        run.summary["exhaustive_selected_families"] = exhaustive_selected
        run.summary["fitness_gap_to_optimum"] = comparison_row["fitness_gap_to_optimum"]
        run.summary["macro_f1_gap_to_optimum"] = comparison_row["macro_f1_gap_to_optimum"]
        run.summary["accuracy_gap_to_optimum"] = comparison_row["accuracy_gap_to_optimum"]
        run.summary["ga_rank_in_exhaustive"] = comparison_row["ga_rank_in_exhaustive"]
        run.summary["total_elapsed_seconds"] = time.perf_counter() - start_time
        run.finish()

    print(
        f"[done] ga_rank_in_exhaustive={comparison_row['ga_rank_in_exhaustive']} "
        f"fitness_gap={comparison_row['fitness_gap_to_optimum']:.4f} "
        f"exact_match={comparison_row['exact_family_match']}"
    )


if __name__ == "__main__":
    main()
