from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
import wandb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_family_selection.genetic_algorithm.common import artifacts_dir, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill final W&B metadata for GA family-selection runs.")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("feature_family_selection/genetic_algorithm/artifacts"),
    )
    parser.add_argument("--run-name", required=True, type=str)
    parser.add_argument(
        "--variant",
        choices=["with_xxx", "without_xxx"],
        default="without_xxx",
        help="Variant directory under the run artifacts.",
    )
    parser.add_argument("--project", default="ser-feature-family-selection")
    parser.add_argument("--entity", required=True)
    parser.add_argument("--run-ids", nargs="*", default=None)
    parser.add_argument(
        "--skip-history-table",
        action="store_true",
        help="Only repair run state and summary metrics without re-logging the final history table.",
    )
    return parser.parse_args()


def discover_run_ids(wandb_dir: Path) -> list[str]:
    run_ids: list[str] = []
    for run_dir in sorted(wandb_dir.glob("run-*/")):
        run_id = run_dir.name.rsplit("-", 1)[-1]
        if run_id:
            run_ids.append(run_id)
    return run_ids


def selected_families_value(best_solution: dict[str, object]) -> list[str]:
    families = best_solution.get("selected_families", [])
    if not isinstance(families, list):
        raise TypeError("Expected 'selected_families' to be a list in best_solution.json")
    return [str(family) for family in families]


def main() -> None:
    args = parse_args()
    include_xxx = args.variant == "with_xxx"
    output_dir = artifacts_dir(args.artifacts_root, args.run_name, include_xxx)
    history_path = output_dir / "generation_history.csv"
    best_solution_path = output_dir / "best_solution.json"
    run_config_path = output_dir / "run_config.json"
    wandb_dir = output_dir / "wandb"

    history_df = pd.read_csv(history_path)
    best_solution = load_json(best_solution_path)
    run_config = load_json(run_config_path)

    run_ids = list(args.run_ids) if args.run_ids else discover_run_ids(wandb_dir)
    if not run_ids:
        raise FileNotFoundError(f"No local W&B run directories found in {wandb_dir}")

    last_generation = int(history_df["generation"].max()) if not history_df.empty else 0
    history_table_step = last_generation + 1
    run_name = f"{args.run_name}_{args.variant}"
    selected_families = selected_families_value(best_solution)

    for run_id in run_ids:
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            id=run_id,
            name=run_name,
            dir=str(output_dir),
            resume="must",
            reinit="create_new",
        )
        if run is None:
            raise RuntimeError(f"wandb.init returned None for run id {run_id}")

        if not args.skip_history_table:
            run.log({"history_table": wandb.Table(dataframe=history_df.copy())}, step=history_table_step)

        run.summary["best_fitness"] = float(best_solution["fitness"])
        run.summary["best_macro_f1"] = float(best_solution["f1_macro"])
        run.summary["best_accuracy"] = float(best_solution["accuracy"])
        run.summary["selected_families"] = selected_families
        run.summary["total_elapsed_seconds"] = float(history_df["total_elapsed_seconds"].max())
        run.summary["final_generation"] = last_generation
        run.summary["num_evaluated_subsets"] = int(history_df["evaluated_subsets"].max())
        run.summary["num_selected_families"] = int(best_solution["num_selected_families"])
        run.summary["validation_mode"] = str(best_solution["validation_mode"])
        run.summary["n_folds"] = int(best_solution["n_folds"])
        run.summary["run_config"] = run_config
        run.finish()

        print(
            f"[backfilled] run_id={run_id} state=finished "
            f"best_macro_f1={best_solution['f1_macro']} selected_families={selected_families}"
        )


if __name__ == "__main__":
    main()
