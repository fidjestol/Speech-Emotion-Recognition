from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import wandb

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_selection.common import DEFAULT_EXCLUDED_EMOTIONS, DEFAULT_REQUIRE_AGREEMENT, variant_name
from feature_family_selection.genetic_algorithm.common import DEFAULT_FAMILY_KEYS, save_json
from feature_family_selection.genetic_algorithm.evaluate_family_subset import evaluate_family_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exhaustive search for feature-family selection.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-name", type=str, default="exhaustive_feature_family_selection")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("feature_family_selection/exhaustive_search/artifacts"),
    )
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILY_KEYS))
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--evaluator-model",
        choices=["random_forest", "logreg", "linear_svc"],
        default="random_forest",
    )
    parser.add_argument(
        "--require-agreement",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REQUIRE_AGREEMENT,
    )
    parser.add_argument(
        "--include-xxx",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Toggle whether rows labeled xxx are kept. Default excludes xxx.",
    )
    parser.add_argument("--excluded-emotions", nargs="+", default=list(DEFAULT_EXCLUDED_EMOTIONS))
    parser.add_argument("--report-to", choices=["wandb", "none"], default="wandb")
    parser.add_argument("--wandb-project", type=str, default="ser-feature-family-selection")
    parser.add_argument("--wandb-entity", type=str, default=None)
    return parser.parse_args()


def artifacts_dir(base_dir: Path, run_name: str, include_xxx: bool) -> Path:
    return base_dir / run_name / variant_name(include_xxx)


def enumerate_family_subsets(families: list[str]) -> Iterable[list[str]]:
    for subset_size in range(1, len(families) + 1):
        for subset in itertools.combinations(families, subset_size):
            yield list(subset)


def init_wandb(args: argparse.Namespace, output_dir: Path):
    if args.report_to != "wandb":
        return None
    total_subsets = (2 ** len(args.families)) - 1
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{args.run_name}_{variant_name(args.include_xxx)}",
        dir=str(output_dir),
        config={
            **vars(args),
            "repo_root": str(args.repo_root),
            "artifacts_root": str(args.artifacts_root),
            "metric": "f1_macro",
            "search_method": "exhaustive",
            "variant": variant_name(args.include_xxx),
            "total_subsets": total_subsets,
        },
    )


def write_outputs(output_dir: Path, args: argparse.Namespace, results_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "all_results.csv", index=False)
    summary_df.to_csv(output_dir / "subset_size_summary.csv", index=False)
    best_row = results_df.iloc[0].to_dict()
    save_json(output_dir / "best_solution.json", best_row)
    save_json(output_dir / "selected_families.json", {"selected_families": best_row["selected_families"]})
    save_json(
        output_dir / "run_config.json",
        {
            **vars(args),
            "repo_root": str(args.repo_root),
            "artifacts_root": str(args.artifacts_root),
        },
    )


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    output_dir = artifacts_dir(args.artifacts_root, args.run_name, args.include_xxx)
    run = init_wandb(args, output_dir)

    total_subsets = (2 ** len(args.families)) - 1
    start_time = time.perf_counter()
    rows: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None

    for evaluated_count, selected_families in enumerate(enumerate_family_subsets(list(args.families)), start=1):
        evaluation = evaluate_family_subset(
            repo_root=args.repo_root,
            family_keys=selected_families,
            include_xxx=args.include_xxx,
            require_agreement=args.require_agreement,
            excluded_emotions=args.excluded_emotions,
            evaluator_model=args.evaluator_model,
            test_size=args.test_size,
            random_state=args.random_state,
            alpha=args.alpha,
            beta=args.beta,
            family_space_size=len(args.families),
        )
        row = {
            "evaluated_subset_index": evaluated_count,
            "subset_size": evaluation.num_selected_families,
            "fitness": evaluation.fitness,
            "accuracy": evaluation.accuracy,
            "f1_macro": evaluation.f1_macro,
            "f1_weighted": evaluation.f1_weighted,
            "num_features": evaluation.num_features,
            "num_rows": evaluation.num_rows,
            "num_train_rows": evaluation.num_train_rows,
            "num_test_rows": evaluation.num_test_rows,
            "selected_families": evaluation.selected_families,
            "selected_families_text": ",".join(evaluation.selected_families),
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        rows.append(row)

        if best_result is None or row["fitness"] > best_result["fitness"]:
            best_result = row

        if run is not None:
            coverage_fraction = evaluated_count / total_subsets
            wandb.log(
                {
                    "search/evaluated_subsets": evaluated_count,
                    "search/coverage_fraction": coverage_fraction,
                    "search/coverage_percent": 100.0 * coverage_fraction,
                    "search/current_fitness": row["fitness"],
                    "search/current_macro_f1": row["f1_macro"],
                    "search/current_accuracy": row["accuracy"],
                    "search/current_subset_size": row["subset_size"],
                    "search/best_fitness": best_result["fitness"],
                    "search/best_macro_f1": best_result["f1_macro"],
                    "search/best_accuracy": best_result["accuracy"],
                    "search/best_subset_size": best_result["subset_size"],
                    "runtime/total_elapsed_seconds": row["elapsed_seconds"],
                },
                step=evaluated_count,
            )

        if evaluated_count == 1 or evaluated_count % 50 == 0 or evaluated_count == total_subsets:
            print(
                f"[subset] index={evaluated_count}/{total_subsets} "
                f"current_fitness={row['fitness']:.4f} best_fitness={best_result['fitness']:.4f} "
                f"best_selected={best_result['selected_families']}"
            )

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(
        by=["fitness", "f1_macro", "accuracy", "subset_size"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    summary_df = (
        results_df.groupby("subset_size", as_index=False)
        .agg(
            subsets_evaluated=("subset_size", "size"),
            best_fitness=("fitness", "max"),
            avg_fitness=("fitness", "mean"),
            best_macro_f1=("f1_macro", "max"),
            avg_macro_f1=("f1_macro", "mean"),
            best_accuracy=("accuracy", "max"),
            avg_accuracy=("accuracy", "mean"),
        )
        .sort_values("subset_size")
        .reset_index(drop=True)
    )
    write_outputs(output_dir, args, results_df, summary_df)

    if run is not None:
        wandb.log(
            {
                "results_table": wandb.Table(dataframe=results_df),
                "subset_size_summary": wandb.Table(dataframe=summary_df),
            }
        )
        best_row = results_df.iloc[0]
        wandb.summary["best_fitness"] = float(best_row["fitness"])
        wandb.summary["best_macro_f1"] = float(best_row["f1_macro"])
        wandb.summary["best_accuracy"] = float(best_row["accuracy"])
        wandb.summary["selected_families"] = list(best_row["selected_families"])
        wandb.summary["subset_size"] = int(best_row["subset_size"])
        wandb.summary["total_subsets"] = total_subsets
        wandb.summary["total_elapsed_seconds"] = time.perf_counter() - start_time
        wandb.finish()

    best_row = results_df.iloc[0]
    print(
        f"[done] best_fitness={best_row['fitness']:.4f} best_macro_f1={best_row['f1_macro']:.4f} "
        f"selected_families={best_row['selected_families']}"
    )


if __name__ == "__main__":
    main()
