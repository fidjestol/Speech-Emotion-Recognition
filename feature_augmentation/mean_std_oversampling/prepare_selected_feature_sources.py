from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_family_selection.genetic_algorithm.common import merge_selected_families
from feature_selection.common import (
    DEFAULT_EXCLUDED_EMOTIONS,
    DEFAULT_REQUIRE_AGREEMENT,
    METADATA_CANDIDATES,
    filter_feature_frame,
    resolve_feature_source,
)

ANOVA_DATASET_KEYS = [
    "bert",
    "mfcc_raw",
    "prosody_energy",
    "prosody_pitch",
    "ssl_hubert",
    "ssl_wav2vec",
    "tfidf",
]

PCA_DATASET_KEYS = [
    "bert",
    "mfcc_raw",
    "prosody_energy",
    "prosody_pitch",
    "ssl_hubert",
    "ssl_wav2vec",
    "tfidf",
]


@dataclass(frozen=True)
class GeneratedSource:
    dataset_key: str
    method: str
    source_dataset_key: str
    output_csv: str
    rows: int
    feature_columns: int
    target_col: str
    session_col: str
    source_path: str
    provenance: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare augmentation-ready CSVs from ANOVA/PCA/GA selection artifacts.")
    parser.add_argument("--bundle-name", required=True, help="Output folder name under feature_augmentation/selected_feature_sources/")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["anova", "pca", "ga"],
        choices=["anova", "pca", "ga"],
        help="Selection methods to materialize.",
    )
    parser.add_argument(
        "--include-xxx",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build sources from the xxx-inclusive variant. Default is without_xxx.",
    )
    parser.add_argument(
        "--require-agreement",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REQUIRE_AGREEMENT,
    )
    parser.add_argument("--excluded-emotions", nargs="*", default=list(DEFAULT_EXCLUDED_EMOTIONS))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def is_metadata_column(column_name: str) -> bool:
    if column_name in METADATA_CANDIDATES:
        return True
    if "__" in column_name:
        suffix = column_name.rsplit("__", 1)[-1]
        return suffix in METADATA_CANDIDATES
    return False


def find_column(columns: list[str], target_name: str) -> str | None:
    if target_name in columns:
        return target_name
    suffix_matches = [column for column in columns if column.rsplit("__", 1)[-1] == target_name]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_selected_feature_names(path: Path) -> list[str]:
    payload = load_json(path)
    if isinstance(payload, dict):
        if "selected_features" in payload:
            payload = payload["selected_features"]
        elif "features" in payload:
            payload = payload["features"]
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of selected features in {path}")
    return [str(value) for value in payload]


def load_filtered_source_frame(
    dataset_key: str,
    *,
    include_xxx: bool,
    require_agreement: bool,
    excluded_emotions: list[str],
) -> tuple[pd.DataFrame, str, str, Path]:
    source_path = resolve_feature_source(REPO_ROOT, dataset_key)
    raw_df = pd.read_csv(source_path)
    target_col = find_column(list(raw_df.columns), "emotion")
    session_col = find_column(list(raw_df.columns), "session")
    if target_col is None:
        raise KeyError(f"Could not find target column 'emotion' in {source_path}")
    if session_col is None:
        raise KeyError(f"Could not find session column 'session' in {source_path}")
    filtered = filter_feature_frame(
        raw_df,
        target_col=target_col,
        include_xxx=include_xxx,
        require_agreement=require_agreement,
        excluded_emotions=excluded_emotions,
    )
    return filtered, target_col, session_col, source_path


def build_model_ready_frame(df: pd.DataFrame, *, target_col: str, session_col: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    metadata_cols = [column for column in df.columns if is_metadata_column(column)]
    if target_col not in metadata_cols:
        metadata_cols.append(target_col)
    if session_col not in metadata_cols:
        metadata_cols.append(session_col)

    feature_cols = [column for column in df.columns if column not in metadata_cols]
    work_df = df[metadata_cols + feature_cols].copy()
    work_df = work_df.dropna(subset=[target_col, session_col]).copy()
    usable_feature_cols = [column for column in feature_cols if work_df[column].notna().all()]
    work_df = work_df[metadata_cols + usable_feature_cols].dropna(axis=0, how="any").copy()
    work_df[session_col] = work_df[session_col].astype(int)
    return work_df.reset_index(drop=True), metadata_cols, usable_feature_cols


def write_generated_source(
    *,
    out_dir: Path,
    dataset_key: str,
    frame: pd.DataFrame,
    metadata_cols: list[str],
    target_col: str,
    session_col: str,
    method: str,
    source_dataset_key: str,
    source_path: Path,
    provenance: dict[str, Any],
) -> GeneratedSource:
    output_path = out_dir / f"{dataset_key}.csv"
    frame.to_csv(output_path, index=False)
    feature_columns = len([column for column in frame.columns if column not in metadata_cols])
    return GeneratedSource(
        dataset_key=dataset_key,
        method=method,
        source_dataset_key=source_dataset_key,
        output_csv=str(output_path),
        rows=int(len(frame)),
        feature_columns=int(feature_columns),
        target_col=target_col,
        session_col=session_col,
        source_path=str(source_path),
        provenance=provenance,
    )


def build_anova_source(
    dataset_key: str,
    *,
    out_dir: Path,
    include_xxx: bool,
    require_agreement: bool,
    excluded_emotions: list[str],
) -> GeneratedSource:
    artifact_dir = REPO_ROOT / "feature_selection" / "anova" / "select_k_best_anova" / "artifacts"
    selected_feature_path = artifact_dir / f"{dataset_key}_selected_features.json"
    run_meta_path = artifact_dir / f"{dataset_key}_run_metadata.json"
    selected_features = load_selected_feature_names(selected_feature_path)
    run_meta = load_json(run_meta_path)

    filtered_df, target_col, session_col, source_path = load_filtered_source_frame(
        dataset_key,
        include_xxx=include_xxx,
        require_agreement=require_agreement,
        excluded_emotions=excluded_emotions,
    )
    work_df, metadata_cols, usable_feature_cols = build_model_ready_frame(
        filtered_df,
        target_col=target_col,
        session_col=session_col,
    )
    selected_feature_set = [column for column in selected_features if column in usable_feature_cols]
    if not selected_feature_set:
        raise RuntimeError(f"No ANOVA-selected columns resolved for dataset {dataset_key}")
    final_df = work_df[metadata_cols + selected_feature_set].copy()
    return write_generated_source(
        out_dir=out_dir,
        dataset_key=f"anova_{dataset_key}",
        frame=final_df,
        metadata_cols=metadata_cols,
        target_col=target_col,
        session_col=session_col,
        method="anova",
        source_dataset_key=dataset_key,
        source_path=source_path,
        provenance={
            "artifact_dir": str(artifact_dir),
            "selected_feature_path": str(selected_feature_path),
            "run_metadata_path": str(run_meta_path),
            "selected_feature_count_requested": len(selected_features),
            "selected_feature_count_resolved": len(selected_feature_set),
            "run_metadata": run_meta,
        },
    )


def build_pca_source(
    dataset_key: str,
    *,
    out_dir: Path,
    include_xxx: bool,
    require_agreement: bool,
    excluded_emotions: list[str],
    random_state: int,
) -> GeneratedSource:
    artifact_dir = REPO_ROOT / "feature_selection" / "pca" / "select_pca" / "artifacts"
    run_meta_path = artifact_dir / f"{dataset_key}_run_metadata.json"
    run_meta = load_json(run_meta_path)

    filtered_df, target_col, session_col, source_path = load_filtered_source_frame(
        dataset_key,
        include_xxx=include_xxx,
        require_agreement=require_agreement,
        excluded_emotions=excluded_emotions,
    )
    work_df, metadata_cols, usable_feature_cols = build_model_ready_frame(
        filtered_df,
        target_col=target_col,
        session_col=session_col,
    )
    X = work_df[usable_feature_cols].astype(np.float32)

    decomposition_method = str(run_meta.get("decomposition_method", "truncated_svd" if dataset_key == "tfidf" else "pca")).strip().lower()
    requested_n_components = int(run_meta["pca_components_effective"])
    if decomposition_method == "truncated_svd":
        max_supported_components = max(1, X.shape[1] - 1)
        n_components = min(requested_n_components, max_supported_components)
        decomposer = TruncatedSVD(
            n_components=n_components,
            algorithm="randomized",
            n_iter=7,
            random_state=random_state,
        )
        transformed = decomposer.fit_transform(X)
    elif decomposition_method == "pca":
        max_supported_components = min(X.shape[0], X.shape[1])
        n_components = min(requested_n_components, max_supported_components)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        decomposer = PCA(
            n_components=n_components,
            random_state=random_state,
            svd_solver="randomized",
        )
        transformed = decomposer.fit_transform(X_scaled)
    else:
        raise ValueError(f"Unsupported decomposition method '{decomposition_method}' for dataset {dataset_key}")

    component_names = [f"pca_{index:04d}" for index in range(transformed.shape[1])]
    transformed_df = pd.DataFrame(transformed, columns=component_names, index=work_df.index)
    final_df = pd.concat([work_df[metadata_cols].reset_index(drop=True), transformed_df.reset_index(drop=True)], axis=1)
    return write_generated_source(
        out_dir=out_dir,
        dataset_key=f"pca_{dataset_key}",
        frame=final_df,
        metadata_cols=metadata_cols,
        target_col=target_col,
        session_col=session_col,
        method="pca",
        source_dataset_key=dataset_key,
        source_path=source_path,
        provenance={
            "artifact_dir": str(artifact_dir),
            "run_metadata_path": str(run_meta_path),
            "decomposition_method": decomposition_method,
            "pca_components_requested": requested_n_components,
            "pca_components_effective": int(transformed.shape[1]),
            "pca_components_max_supported": int(max_supported_components),
            "run_metadata": run_meta,
        },
    )


def build_ga_source(
    *,
    out_dir: Path,
    include_xxx: bool,
    require_agreement: bool,
    excluded_emotions: list[str],
) -> GeneratedSource:
    artifact_dir = REPO_ROOT / "feature_selection" / "ga" / "artifacts" / "ga_subfamily_from_ga_family_loso_full_v1" / "without_xxx"
    best_solution_path = artifact_dir / "best_solution.json"
    selected_feature_path = artifact_dir / "selected_features.json"
    best_solution = load_json(best_solution_path)
    selected_features = load_selected_feature_names(selected_feature_path)
    selected_families = [str(value) for value in best_solution["selected_families"]]
    merged = merge_selected_families(
        REPO_ROOT,
        selected_families,
        include_xxx=include_xxx,
        require_agreement=require_agreement,
        excluded_emotions=excluded_emotions,
    )
    target_col = find_column(list(merged.columns), "emotion")
    session_col = find_column(list(merged.columns), "session")
    if target_col is None or session_col is None:
        raise KeyError("Merged GA source frame is missing required metadata columns.")
    metadata_cols = [column for column in merged.columns if is_metadata_column(column)]
    selected_feature_set = [column for column in selected_features if column in merged.columns]
    if not selected_feature_set:
        raise RuntimeError("No GA-selected columns resolved after family merge.")
    final_df = merged[metadata_cols + selected_feature_set].copy().reset_index(drop=True)
    return write_generated_source(
        out_dir=out_dir,
        dataset_key="ga_subfamily_from_family_loso_full_v1",
        frame=final_df,
        metadata_cols=metadata_cols,
        target_col=target_col,
        session_col=session_col,
        method="ga",
        source_dataset_key="ga_subfamily_from_family_loso_full_v1",
        source_path=REPO_ROOT / "extracted_features" / "combined" / "all_families_features.csv",
        provenance={
            "artifact_dir": str(artifact_dir),
            "best_solution_path": str(best_solution_path),
            "selected_feature_path": str(selected_feature_path),
            "selected_family_count": len(selected_families),
            "selected_feature_count_requested": len(selected_features),
            "selected_feature_count_resolved": len(selected_feature_set),
            "best_solution": best_solution,
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = REPO_ROOT / "feature_augmentation" / "selected_feature_sources" / args.bundle_name
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[GeneratedSource] = []
    if "anova" in args.methods:
        for dataset_key in ANOVA_DATASET_KEYS:
            generated.append(
                build_anova_source(
                    dataset_key,
                    out_dir=out_dir,
                    include_xxx=args.include_xxx,
                    require_agreement=args.require_agreement,
                    excluded_emotions=args.excluded_emotions,
                )
            )
    if "pca" in args.methods:
        for dataset_key in PCA_DATASET_KEYS:
            generated.append(
                build_pca_source(
                    dataset_key,
                    out_dir=out_dir,
                    include_xxx=args.include_xxx,
                    require_agreement=args.require_agreement,
                    excluded_emotions=args.excluded_emotions,
                    random_state=args.random_state,
                )
            )
    if "ga" in args.methods:
        generated.append(
            build_ga_source(
                out_dir=out_dir,
                include_xxx=args.include_xxx,
                require_agreement=args.require_agreement,
                excluded_emotions=args.excluded_emotions,
            )
        )

    manifest = {
        "bundle_name": args.bundle_name,
        "include_xxx": bool(args.include_xxx),
        "require_agreement": bool(args.require_agreement),
        "excluded_emotions": list(args.excluded_emotions),
        "datasets": {
            item.dataset_key: {
                "path": Path(item.output_csv).name,
                "method": item.method,
                "source_dataset_key": item.source_dataset_key,
                "rows": item.rows,
                "feature_columns": item.feature_columns,
                "target_col": item.target_col,
                "session_col": item.session_col,
                "source_path": item.source_path,
                "provenance": item.provenance,
            }
            for item in generated
        },
        "summary": [asdict(item) for item in generated],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame([asdict(item) for item in generated]).to_csv(out_dir / "summary.csv", index=False)

    print(f"Wrote {len(generated)} generated sources under {out_dir}")
    print(f"Manifest: {manifest_path}")
    for item in generated:
        print(f" - {item.dataset_key}: rows={item.rows} feature_columns={item.feature_columns} method={item.method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
