from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_augmentation.common import resolve_augmentation_artifact_dir  # noqa: E402
from feature_augmentation.mean_std_oversampling.run_augmentation_experiment import (  # noqa: E402
    ALL_MODEL_NAMES,
    environment_snapshot,
)
from feature_selection.common import (  # noqa: E402
    DEFAULT_REQUIRE_AGREEMENT,
    FEATURE_SOURCE_RELATIVE_PATHS,
    resolve_feature_source,
    variant_name,
)
from utils.wandb_multi import init_multi_wandb_run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all feature augmentation experiments.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset-keys", nargs="*", default=sorted(FEATURE_SOURCE_RELATIVE_PATHS))
    parser.add_argument("--run-both-variants", action="store_true")
    parser.add_argument("--include-xxx", action="store_true")
    parser.add_argument("--use-variant-dirs", action="store_true")
    parser.add_argument(
        "--require-agreement",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REQUIRE_AGREEMENT,
    )
    parser.add_argument("--excluded-emotions", nargs="*", default=["oth", "dis"])
    parser.add_argument("--target-per-class", type=int, default=10_000)
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-nrows", type=int)
    parser.add_argument(
        "--enable-gpu-models",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--wandb-project", default="ser-feature-augmentation")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip dataset/variant outputs that already have run metadata. Default is to resume existing runs.",
    )
    parser.add_argument(
        "--save-augmented-matrices",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--save-models",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--train-augmented-final-stacking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to train the exported augmented_final stacking model. "
            "This does not affect cross-validation metrics, only the final saved model."
        ),
    )
    parser.add_argument("--models", nargs="*", default=list(ALL_MODEL_NAMES), choices=ALL_MODEL_NAMES)
    return parser.parse_args(argv)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def runner_wandb_state_path(run_root: Path) -> Path:
    return run_root / "runner_wandb_run.json"


def load_saved_runner_wandb_id(run_root: Path) -> str | None:
    path = runner_wandb_state_path(run_root)
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    run_id = str(payload.get("id", "")).strip()
    return run_id or None


def save_runner_wandb_state(run_root: Path, run: Any) -> None:
    write_json(
        runner_wandb_state_path(run_root),
        {
            "id": str(getattr(run, "id", "")),
            "name": str(getattr(run, "name", "")),
            "url": getattr(run, "url", None),
            "project": getattr(run, "project", None),
            "entity": getattr(run, "entity", None),
        },
    )


def persist_runner_state(run_root: Path, results: list[dict[str, Any]], failures: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_df = pd.DataFrame(results)
    failures_df = pd.DataFrame(failures)
    results_df.to_csv(run_root / "run_manifest.csv", index=False)
    write_json(run_root / "run_manifest.json", {"results": results})
    failures_df.to_csv(run_root / "failed_runs.csv", index=False)
    write_json(run_root / "failed_runs.json", {"failures": failures})

    if not results_df.empty:
        summary_columns = [
            "dataset_key",
            "variant",
            "status",
            "seconds",
            "best_baseline_model",
            "best_augmented_model",
            "baseline_best_accuracy",
            "augmented_best_accuracy",
            "baseline_best_f1_weighted",
            "augmented_best_f1_weighted",
            "baseline_best_f1_macro",
            "augmented_best_f1_macro",
        ]
        available_columns = [column for column in summary_columns if column in results_df.columns]
        results_df[available_columns].to_csv(run_root / "results_summary.csv", index=False)

    return results_df, failures_df


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    variants = [False, True] if args.run_both_variants else [bool(args.include_xxx)]

    run_root = REPO_ROOT / "feature_augmentation" / "mean_std_oversampling" / "artifacts" / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(
        run_root / "environment_summary.json",
        {
            **environment_snapshot(),
            "hostname": socket.gethostname(),
            "run_name": args.run_name,
            "dataset_keys": args.dataset_keys,
            "variants": [variant_name(flag) for flag in variants],
            "report_to": args.report_to,
        },
    )

    runner_wandb = None
    if "wandb" in {part.strip() for part in args.report_to.split(",") if part.strip()}:
        init_kwargs = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "group": args.wandb_group or args.run_name,
            "job_type": "feature_augmentation_runner",
            "name": f"{args.run_name}_runner",
            "config": vars(args),
            "reinit": True,
        }
        saved_run_id = load_saved_runner_wandb_id(run_root)
        if saved_run_id is not None:
            init_kwargs["id"] = saved_run_id
            init_kwargs["resume"] = "allow"
        runner_wandb = init_multi_wandb_run(**init_kwargs)
        save_runner_wandb_state(run_root, runner_wandb)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for include_xxx in variants:
        current_variant = variant_name(include_xxx)
        for dataset_key in args.dataset_keys:
            source_path = resolve_feature_source(REPO_ROOT, dataset_key)
            artifact_dir = resolve_augmentation_artifact_dir(
                REPO_ROOT,
                include_xxx=include_xxx,
                use_variant_dirs=args.use_variant_dirs,
                run_name=args.run_name,
                dataset_key=dataset_key,
            )
            meta_path = artifact_dir / f"{dataset_key}_run_metadata.json"
            if args.skip_existing and meta_path.exists():
                row = {
                    "dataset_key": dataset_key,
                    "variant": current_variant,
                    "status": "skipped_existing",
                    "seconds": 0.0,
                    "source_path": str(source_path),
                    "artifact_dir": str(artifact_dir),
                    "error": None,
                }
                results.append(row)
                persist_runner_state(run_root, results, failures)
                continue

            cmd = [
                sys.executable,
                str(REPO_ROOT / "feature_augmentation" / "mean_std_oversampling" / "run_augmentation_experiment.py"),
                "--dataset-key",
                dataset_key,
                "--run-name",
                args.run_name,
                "--target-per-class",
                str(args.target_per_class),
                "--group-size",
                str(args.group_size),
                "--random-state",
                str(args.random_state),
                "--augmentation-random-state",
                str(args.random_state),
                "--report-to",
                args.report_to,
                "--wandb-project",
                args.wandb_project,
                "--excluded-emotions",
                *args.excluded_emotions,
                "--models",
                *args.models,
            ]
            if args.use_variant_dirs:
                cmd.append("--use-variant-dirs")
            if include_xxx:
                cmd.append("--include-xxx")
            if args.require_agreement:
                cmd.append("--require-agreement")
            else:
                cmd.append("--no-require-agreement")
            if args.use_nrows is not None:
                cmd.extend(["--use-nrows", str(args.use_nrows)])
            if args.enable_gpu_models:
                cmd.append("--enable-gpu-models")
            else:
                cmd.append("--no-enable-gpu-models")
            if args.save_augmented_matrices:
                cmd.append("--save-augmented-matrices")
            else:
                cmd.append("--no-save-augmented-matrices")
            if args.save_models:
                cmd.append("--save-models")
            else:
                cmd.append("--no-save-models")
            if args.train_augmented_final_stacking:
                cmd.append("--train-augmented-final-stacking")
            else:
                cmd.append("--no-train-augmented-final-stacking")
            if args.wandb_entity:
                cmd.extend(["--wandb-entity", args.wandb_entity])
            if args.wandb_group:
                cmd.extend(["--wandb-group", args.wandb_group])

            started = time.perf_counter()
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                    env=os.environ.copy(),
                )
                elapsed = round(time.perf_counter() - started, 3)
                stdout = completed.stdout.strip()
                stderr = completed.stderr.strip()
                try:
                    payload = json.loads(stdout.splitlines()[-1] if stdout else "{}")
                except json.JSONDecodeError:
                    payload = {}
                row = {
                    "dataset_key": dataset_key,
                    "variant": current_variant,
                    "status": "ok" if completed.returncode == 0 else "failed",
                    "seconds": elapsed,
                    "source_path": str(source_path),
                    "artifact_dir": str(artifact_dir),
                    "error": None if completed.returncode == 0 else (payload.get("error") or stderr[-1000:]),
                    "stdout_log": stdout[-4000:],
                    "stderr_log": stderr[-4000:],
                }
                row.update({k: v for k, v in payload.items() if k not in row})
                results.append(row)
                if completed.returncode != 0:
                    failure_payload = {
                        **row,
                        "traceback": payload.get("traceback", stderr[-8000:]),
                    }
                    failures.append(failure_payload)
                persist_runner_state(run_root, results, failures)
                if runner_wandb is not None:
                    import wandb

                    runner_wandb.log(
                        {
                            "progress/completed": len(results),
                            "progress/failed": len(failures),
                            f"{current_variant}/{dataset_key}/seconds": elapsed,
                            f"{current_variant}/{dataset_key}/status": 1 if completed.returncode == 0 else 0,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                elapsed = round(time.perf_counter() - started, 3)
                failure_payload = {
                    "dataset_key": dataset_key,
                    "variant": current_variant,
                    "status": "failed",
                    "seconds": elapsed,
                    "source_path": str(source_path),
                    "artifact_dir": str(artifact_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                results.append(failure_payload)
                failures.append(failure_payload)
                persist_runner_state(run_root, results, failures)

    results_df, failures_df = persist_runner_state(run_root, results, failures)

    latest_run_path = REPO_ROOT / "feature_augmentation" / "mean_std_oversampling" / "artifacts" / "latest_run.txt"
    latest_run_path.write_text(str(run_root), encoding="utf-8")

    if runner_wandb is not None:
        import wandb

        runner_wandb.log(
            {
                "run_manifest": wandb.Table(dataframe=results_df),
                "failed_runs": wandb.Table(dataframe=failures_df if not failures_df.empty else pd.DataFrame(columns=["dataset_key", "variant"])),
            }
        )
        runner_wandb.summary["completed_count"] = int((results_df["status"] == "ok").sum()) if not results_df.empty else 0
        runner_wandb.summary["failed_count"] = int((results_df["status"] == "failed").sum()) if not results_df.empty else 0
        runner_wandb.finish()

    print(results_df.to_string(index=False) if not results_df.empty else "No runs executed.")
    if failures:
        print("\nFailures saved to:", run_root / "failed_runs.csv")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
