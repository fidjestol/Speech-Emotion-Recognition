# Exhaustive Search for Feature Family Selection

This folder provides a brute-force alternative to the genetic algorithm.

Instead of sampling candidate family subsets heuristically, exhaustive search
evaluates every non-empty subset of the configured feature families and returns
the best one under the same evaluation pipeline and fitness formula used by the
GA implementation.

Why this is useful here:

- the default family space has 10 families, so there are only `2^10 - 1 = 1023`
  non-empty subsets
- it gives an exact optimum for the current search space
- it provides a clean reference point for judging how well the GA performs

## Output Artifacts

The exhaustive script saves:

- `all_results.csv`: one row per evaluated subset
- `best_solution.json`: the top-ranked subset and its metrics
- `selected_families.json`: the best subset's family names
- `run_config.json`: the configuration used for the run
- `subset_size_summary.csv`: aggregate metrics by subset size

If W&B reporting is enabled, it also logs:

- progress through the search space
- current best fitness / macro F1 / accuracy
- a final `results_table`
- a final `subset_size_summary` table

W&B reporting is enabled by default through `--report-to wandb`.

## Compare Against GA

After you have both a GA run and an exhaustive run, you can compare them with:

```bash
.venv/bin/python feature_family_selection/exhaustive_search/compare_with_ga.py \
  --ga-run-name ga_feature_family_selection_without_xxx_v1 \
  --exhaustive-run-name exhaustive_feature_family_selection
```

This writes comparison artifacts under
`feature_family_selection/exhaustive_search/comparisons/...` and, by default,
logs the comparison summary and exhaustive top-k table to W&B.

## Example

```bash
.venv/bin/python feature_family_selection/exhaustive_search/exhaustive_family_selection.py \
  --run-name exhaustive_feature_family_selection
```
