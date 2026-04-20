from __future__ import annotations

import argparse
import csv
import pickle
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import wandb

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_selection.common import DEFAULT_EXCLUDED_EMOTIONS, DEFAULT_REQUIRE_AGREEMENT, variant_name
from feature_family_selection.genetic_algorithm.common import (
    DEFAULT_CHECKPOINT_BASENAME,
    DEFAULT_FAMILY_KEYS,
    artifacts_dir,
    chromosome_to_string,
    ensure_non_empty_chromosome,
    family_keys_from_chromosome,
    save_json,
    serializable_history,
)
from feature_family_selection.genetic_algorithm.evaluate_family_subset import evaluate_family_subset
from utils.wandb_multi import init_multi_wandb_run


@dataclass
class CandidateResult:
    chromosome: list[int]
    chromosome_key: str
    fitness: float
    accuracy: float
    f1_macro: float
    f1_weighted: float
    num_selected_families: int
    num_features: int
    num_rows: int
    num_train_rows: int
    num_test_rows: int
    selected_families: list[str]

    @classmethod
    def from_eval(cls, chromosome: list[int], evaluation: dict[str, Any]) -> "CandidateResult":
        payload = dict(evaluation)
        return cls(
            chromosome=list(chromosome),
            chromosome_key=chromosome_to_string(chromosome),
            fitness=float(payload["fitness"]),
            accuracy=float(payload["accuracy"]),
            f1_macro=float(payload["f1_macro"]),
            f1_weighted=float(payload["f1_weighted"]),
            num_selected_families=int(payload["num_selected_families"]),
            num_features=int(payload["num_features"]),
            num_rows=int(payload["num_rows"]),
            num_train_rows=int(payload["num_train_rows"]),
            num_test_rows=int(payload["num_test_rows"]),
            selected_families=list(payload["selected_families"]),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genetic algorithm for feature-family selection.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-name", type=str, default="ga_feature_family_selection")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("feature_family_selection/genetic_algorithm/artifacts"),
    )
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILY_KEYS))
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--elitism-k", type=int, default=3)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--mutation-rate", type=float, default=None)
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--min-selected-families", type=int, default=1)
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
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report-to", choices=["wandb", "none"], default="wandb")
    parser.add_argument("--wandb-project", type=str, default="ser-feature-family-selection")
    parser.add_argument("--wandb-entity", type=str, default=None)
    return parser.parse_args()


def random_chromosome(num_families: int, rng: random.Random, min_selected: int) -> list[int]:
    chromosome = [1 if rng.random() < 0.5 else 0 for _ in range(num_families)]
    return ensure_non_empty_chromosome(chromosome, rng, min_selected=min_selected)


def seeded_population(args: argparse.Namespace, rng: random.Random) -> list[list[int]]:
    family_count = len(args.families)
    seeds: list[list[int]] = []
    seeds.append([1] * family_count)
    for idx in range(family_count):
        chrom = [0] * family_count
        chrom[idx] = 1
        seeds.append(chrom)
    for density in (0.3, 0.5, 0.7):
        for _ in range(3):
            chrom = [1 if rng.random() < density else 0 for _ in range(family_count)]
            seeds.append(ensure_non_empty_chromosome(chrom, rng, min_selected=args.min_selected_families))
    while len(seeds) < args.population_size:
        seeds.append(random_chromosome(family_count, rng, args.min_selected_families))
    return seeds[: args.population_size]


def tournament_select(
    population: Sequence[list[int]],
    scores: Sequence[float],
    rng: random.Random,
    tournament_size: int,
) -> list[int]:
    indices = rng.sample(range(len(population)), k=min(tournament_size, len(population)))
    winner = max(indices, key=lambda idx: scores[idx])
    return list(population[winner])


def uniform_crossover(
    parent_a: list[int],
    parent_b: list[int],
    rng: random.Random,
    crossover_rate: float,
) -> tuple[list[int], list[int]]:
    if rng.random() > crossover_rate:
        return list(parent_a), list(parent_b)
    child_a, child_b = [], []
    for bit_a, bit_b in zip(parent_a, parent_b):
        if rng.random() < 0.5:
            child_a.append(bit_a)
            child_b.append(bit_b)
        else:
            child_a.append(bit_b)
            child_b.append(bit_a)
    return child_a, child_b


def mutate(
    chromosome: list[int],
    rng: random.Random,
    mutation_rate: float,
    min_selected: int,
) -> list[int]:
    for idx in range(len(chromosome)):
        if rng.random() < mutation_rate:
            chromosome[idx] = 1 - chromosome[idx]
    return ensure_non_empty_chromosome(chromosome, rng, min_selected=min_selected)


def evaluate_population(
    population: Sequence[list[int]],
    *,
    args: argparse.Namespace,
    eval_cache: dict[str, dict[str, Any]],
) -> list[CandidateResult]:
    results: list[CandidateResult] = []
    for chromosome in population:
        key = chromosome_to_string(chromosome)
        if key not in eval_cache:
            selected_families = family_keys_from_chromosome(chromosome, args.families)
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
            ).to_dict()
            eval_cache[key] = evaluation
        results.append(CandidateResult.from_eval(chromosome, eval_cache[key]))
    return results


def checkpoint_paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / f"{DEFAULT_CHECKPOINT_BASENAME}.pkl", output_dir / f"{DEFAULT_CHECKPOINT_BASENAME}.json"


def save_checkpoint(
    output_dir: Path,
    *,
    next_generation: int,
    population: Sequence[list[int]],
    history: Sequence[dict[str, Any]],
    best_result: CandidateResult,
    eval_cache: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    pkl_path, json_path = checkpoint_paths(output_dir)
    payload = {
        "next_generation": next_generation,
        "population": [list(chrom) for chrom in population],
        "history": list(history),
        "best_result": asdict(best_result),
        "eval_cache": eval_cache,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
    }
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as handle:
        pickle.dump(payload, handle)
    save_json(
        json_path,
        {
            "next_generation": next_generation,
            "best_result": asdict(best_result),
            "history": serializable_history(history),
            "variant": variant_name(args.include_xxx),
            "families": list(args.families),
        },
    )


def load_checkpoint(output_dir: Path) -> dict[str, Any] | None:
    pkl_path, _ = checkpoint_paths(output_dir)
    if not pkl_path.exists():
        return None
    with pkl_path.open("rb") as handle:
        return pickle.load(handle)


def write_generation_rows(output_dir: Path, history: Sequence[dict[str, Any]]) -> None:
    history_path = output_dir / "generation_history.csv"
    if not history:
        return
    fieldnames = list(history[0].keys())
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(serializable_history(history))


def write_best_outputs(
    output_dir: Path,
    best_result: CandidateResult,
    history: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    save_json(output_dir / "best_solution.json", asdict(best_result))
    save_json(output_dir / "selected_families.json", {"selected_families": best_result.selected_families})
    save_json(
        output_dir / "run_config.json",
        {
            **vars(args),
            "repo_root": str(args.repo_root),
            "artifacts_root": str(args.artifacts_root),
        },
    )
    write_generation_rows(output_dir, history)


def log_generation_to_wandb(
    run,
    generation: int,
    results: Sequence[CandidateResult],
    best_result: CandidateResult,
    args: argparse.Namespace,
    search_space_size: int,
    evaluated_subset_count: int,
    generation_runtime_seconds: float,
    total_runtime_seconds: float,
) -> None:
    fitness_values = np.array([result.fitness for result in results], dtype=float)
    macro_values = np.array([result.f1_macro for result in results], dtype=float)
    selected_counts = np.array([result.num_selected_families for result in results], dtype=float)
    unique_chromosomes = len({result.chromosome_key for result in results})
    duplicate_fraction = 1.0 - (unique_chromosomes / max(1, len(results)))
    coverage_fraction = evaluated_subset_count / max(1, search_space_size)
    payload: dict[str, Any] = {
        "generation": generation,
        "generation/best_fitness": float(fitness_values.max()),
        "generation/avg_fitness": float(fitness_values.mean()),
        "generation/worst_fitness": float(fitness_values.min()),
        "generation/best_macro_f1": float(macro_values.max()),
        "generation/avg_macro_f1": float(macro_values.mean()),
        "generation/avg_selected_families": float(selected_counts.mean()),
        "generation/best_selected_families": int(best_result.num_selected_families),
        "generation/unique_chromosomes": unique_chromosomes,
        "generation/duplicate_fraction": float(duplicate_fraction),
        "search/evaluated_subsets": int(evaluated_subset_count),
        "search/coverage_fraction": float(coverage_fraction),
        "search/coverage_percent": float(100.0 * coverage_fraction),
        "runtime/generation_seconds": float(generation_runtime_seconds),
        "runtime/total_elapsed_seconds": float(total_runtime_seconds),
    }
    for family in args.families:
        payload[f"best_family_selected/{family}"] = 1 if family in best_result.selected_families else 0
    run.log(payload, step=generation)


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
            "artifacts_root": str(args.artifacts_root),
            "metric": "f1_macro",
            "variant": variant_name(args.include_xxx),
        },
    )


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    output_dir = artifacts_dir(args.artifacts_root, args.run_name, args.include_xxx)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.random_state)
    np.random.seed(args.random_state)
    rng = random
    mutation_rate = args.mutation_rate if args.mutation_rate is not None else (1.0 / max(1, len(args.families)))
    run_start_time = time.perf_counter()
    search_space_size = (2 ** len(args.families)) - 1

    run = init_wandb(args, output_dir)
    checkpoint = load_checkpoint(output_dir) if args.resume else None
    if checkpoint:
        population = [list(chrom) for chrom in checkpoint["population"]]
        history = list(checkpoint["history"])
        best_result = CandidateResult(**checkpoint["best_result"])
        eval_cache = dict(checkpoint["eval_cache"])
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        start_generation = int(checkpoint["next_generation"])
        print(f"[resume] loaded checkpoint from generation={start_generation}")
    else:
        population = seeded_population(args, rng)
        history: list[dict[str, Any]] = []
        eval_cache: dict[str, dict[str, Any]] = {}
        results = evaluate_population(population, args=args, eval_cache=eval_cache)
        best_result = max(results, key=lambda result: result.fitness)
        start_generation = 0

    for generation in range(start_generation, args.generations):
        generation_start_time = time.perf_counter()
        results = evaluate_population(population, args=args, eval_cache=eval_cache)
        scores = [result.fitness for result in results]
        generation_best = max(results, key=lambda result: result.fitness)
        if generation_best.fitness >= best_result.fitness:
            best_result = generation_best
        generation_runtime_seconds = time.perf_counter() - generation_start_time
        total_runtime_seconds = time.perf_counter() - run_start_time
        unique_chromosomes = len({result.chromosome_key for result in results})
        duplicate_fraction = 1.0 - (unique_chromosomes / max(1, len(results)))
        evaluated_subset_count = len(eval_cache)
        coverage_fraction = evaluated_subset_count / max(1, search_space_size)

        history_row = {
            "generation": generation,
            "best_fitness": generation_best.fitness,
            "avg_fitness": float(np.mean(scores)),
            "worst_fitness": float(np.min(scores)),
            "best_macro_f1": generation_best.f1_macro,
            "avg_macro_f1": float(np.mean([result.f1_macro for result in results])),
            "best_accuracy": generation_best.accuracy,
            "avg_selected_families": float(np.mean([result.num_selected_families for result in results])),
            "unique_chromosomes": unique_chromosomes,
            "duplicate_fraction": float(duplicate_fraction),
            "evaluated_subsets": int(evaluated_subset_count),
            "coverage_fraction": float(coverage_fraction),
            "coverage_percent": float(100.0 * coverage_fraction),
            "best_chromosome": generation_best.chromosome_key,
            "best_selected_families": ",".join(generation_best.selected_families),
            "generation_runtime_seconds": float(generation_runtime_seconds),
            "total_elapsed_seconds": float(total_runtime_seconds),
        }
        history.append(history_row)

        print(
            f"[generation] index={generation + 1}/{args.generations} best_fitness={generation_best.fitness:.4f} "
            f"best_macro_f1={generation_best.f1_macro:.4f} selected={generation_best.selected_families} "
            f"duplicate_fraction={duplicate_fraction:.3f} generation_seconds={generation_runtime_seconds:.2f} "
            f"total_seconds={total_runtime_seconds:.2f}"
        )
        if run is not None:
            log_generation_to_wandb(
                run,
                generation,
                results,
                generation_best,
                args,
                search_space_size=search_space_size,
                evaluated_subset_count=evaluated_subset_count,
                generation_runtime_seconds=generation_runtime_seconds,
                total_runtime_seconds=total_runtime_seconds,
            )

        ranked = sorted(results, key=lambda result: result.fitness, reverse=True)
        next_population = [list(candidate.chromosome) for candidate in ranked[: args.elitism_k]]
        while len(next_population) < args.population_size:
            parent_a = tournament_select(population, scores, rng, args.tournament_size)
            parent_b = tournament_select(population, scores, rng, args.tournament_size)
            child_a, child_b = uniform_crossover(parent_a, parent_b, rng, args.crossover_rate)
            child_a = mutate(child_a, rng, mutation_rate, args.min_selected_families)
            child_b = mutate(child_b, rng, mutation_rate, args.min_selected_families)
            next_population.append(child_a)
            if len(next_population) < args.population_size:
                next_population.append(child_b)

        population = next_population[: args.population_size]
        save_checkpoint(
            output_dir,
            next_generation=generation + 1,
            population=population,
            history=history,
            best_result=best_result,
            eval_cache=eval_cache,
            args=args,
        )
        write_best_outputs(output_dir, best_result, history, args)

    if run is not None:
        history_df = pd.DataFrame(history)
        run.log({"history_table": wandb.Table(dataframe=history_df)})
        run.summary["best_fitness"] = best_result.fitness
        run.summary["best_macro_f1"] = best_result.f1_macro
        run.summary["best_accuracy"] = best_result.accuracy
        run.summary["selected_families"] = best_result.selected_families
        run.summary["total_elapsed_seconds"] = time.perf_counter() - run_start_time
        run.finish()

    print(
        f"[done] best_fitness={best_result.fitness:.4f} best_macro_f1={best_result.f1_macro:.4f} "
        f"selected_families={best_result.selected_families}"
    )


if __name__ == "__main__":
    main()
