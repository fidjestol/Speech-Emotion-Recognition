from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import json
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_MODEL_ID = "iic/emotion2vec_base"
DEFAULT_DATASET_CSV = "datasets/IEMOCAP/iemocap_full_dataset.csv"
DEFAULT_AUDIO_ROOT = "datasets/IEMOCAP"
DEFAULT_OUTPUT_ROOT = "extracted_features/emotion2vec_base/frame"
DEFAULT_MANIFEST_PATH = "extracted_features/emotion2vec_base/frame_manifest.csv"
DEFAULT_SUMMARY_PATH = "extracted_features/emotion2vec_base/frame_summary.json"
DEFAULT_EXCLUDED_EMOTIONS = ("xxx",)


@dataclass(frozen=True)
class ExtractionRecord:
    sample_id: str
    label: str
    agreement: int
    audio_relpath: str
    audio_path: Path
    output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract emotion2vec_base embeddings from IEMOCAP or another CSV-defined audio set "
            "using the Hugging Face / FunASR model."
        )
    )
    parser.add_argument("--dataset-csv", default=DEFAULT_DATASET_CSV)
    parser.add_argument("--audio-root", default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--hub",
        choices=["hf", "huggingface", "ms", "modelscope"],
        default="hf",
        help="FunASR hub selector. Use hf/huggingface for the Hugging Face-hosted model.",
    )
    parser.add_argument("--path-col", default="path")
    parser.add_argument("--label-col", default="emotion")
    parser.add_argument("--agreement-col", default="agreement")
    parser.add_argument("--sample-id-col", default=None)
    parser.add_argument(
        "--granularity",
        choices=["frame", "utterance"],
        default="frame",
        help="The official model supports frame-level or utterance-level embeddings.",
    )
    parser.add_argument(
        "--include-emotions",
        nargs="+",
        default=None,
        help="Optional allow-list. If omitted, every label except excluded ones is kept.",
    )
    parser.add_argument(
        "--exclude-emotions",
        nargs="+",
        default=list(DEFAULT_EXCLUDED_EMOTIONS),
        help="Labels to skip before extraction. Default excludes xxx.",
    )
    parser.add_argument(
        "--min-agreement",
        type=int,
        default=0,
        help="Only keep rows whose agreement is >= this value.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip files that already have a saved .pt embedding.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--funasr-output-dir",
        default=None,
        help="Optional directory passed through to FunASR if you also want its native npy outputs.",
    )
    return parser.parse_args()


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root from the current path.")


def resolve_repo_path(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def normalize_label(label: str) -> str:
    return str(label).strip().lower()


def read_extraction_records(args: argparse.Namespace, repo_root: Path) -> list[ExtractionRecord]:
    dataset_csv = resolve_repo_path(args.dataset_csv, repo_root)
    audio_root = resolve_repo_path(args.audio_root, repo_root)
    output_root = resolve_repo_path(args.output_root, repo_root)

    include_labels = {normalize_label(label) for label in args.include_emotions} if args.include_emotions else None
    exclude_labels = {normalize_label(label) for label in args.exclude_emotions}

    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_csv}")

    records: list[ExtractionRecord] = []
    with dataset_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            label = normalize_label(row[args.label_col])
            if include_labels is not None and label not in include_labels:
                continue
            if label in exclude_labels:
                continue

            agreement = int(float(row.get(args.agreement_col, 0) or 0))
            if agreement < args.min_agreement:
                continue

            audio_relpath = str(row[args.path_col]).replace("\\", "/").strip()
            audio_path = (audio_root / audio_relpath).resolve()
            sample_id = (
                str(row[args.sample_id_col]).strip()
                if args.sample_id_col and row.get(args.sample_id_col)
                else f"{Path(audio_relpath).stem}_{row_idx}"
            )
            output_path = (output_root / Path(audio_relpath)).with_suffix(".pt").resolve()

            records.append(
                ExtractionRecord(
                    sample_id=sample_id,
                    label=label,
                    agreement=agreement,
                    audio_relpath=audio_relpath,
                    audio_path=audio_path,
                    output_path=output_path,
                )
            )

            if args.max_samples is not None and len(records) >= args.max_samples:
                break

    if not records:
        raise RuntimeError("No rows remained after filtering. Check your label/agreement filters.")

    return records


def init_model(args: argparse.Namespace) -> Any:
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "FunASR is required for emotion2vec extraction. "
            "Install it with `bash scripts/setup_emotion2vec_idun_env.sh` on Idun."
        ) from exc

    candidate_model_ids = [args.model_id]
    if args.hub in {"hf", "huggingface"} and args.model_id == "iic/emotion2vec_base":
        candidate_model_ids.append("emotion2vec/emotion2vec_base")
    if args.hub in {"ms", "modelscope"} and args.model_id == "emotion2vec/emotion2vec_base":
        candidate_model_ids.append("iic/emotion2vec_base")

    last_error: Exception | None = None
    for model_id in candidate_model_ids:
        try:
            return AutoModel(model=model_id, hub=args.hub, disable_update=True)
        except Exception as exc:
            last_error = exc
            print(f"[init_model] failed with model_id={model_id} hub={args.hub}: {type(exc).__name__}: {exc}")

    if last_error is not None:
        raise RuntimeError(
            "Failed to initialize emotion2vec via FunASR. "
            "If you are targeting the Hugging Face-hosted model, make sure the active environment has "
            "`funasr`, `huggingface_hub`, and a FunASR build that supports emotion2vec_base."
        ) from last_error
    raise RuntimeError("Failed to initialize emotion2vec via FunASR.")


def run_inference(model: Any, audio_path: Path, args: argparse.Namespace) -> Any:
    """Try the official FunASR calling patterns documented by Hugging Face and the original repo."""
    candidate_methods: list[tuple[str, Any]] = []
    if hasattr(model, "generate"):
        candidate_methods.append(("generate", model.generate))
    candidate_methods.append(("call", model))

    kwargs_variants: list[dict[str, Any]] = [
        {"granularity": args.granularity, "extract_embedding": True},
        {"granularity": args.granularity},
    ]
    if args.funasr_output_dir is not None:
        kwargs_variants = [
            {"output_dir": args.funasr_output_dir, **kwargs}
            for kwargs in kwargs_variants
        ] + kwargs_variants

    last_error: Exception | None = None
    for _, method in candidate_methods:
        for kwargs in kwargs_variants:
            try:
                return method(str(audio_path), **kwargs)
            except TypeError as exc:
                last_error = exc
                continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("Could not find a compatible FunASR calling pattern.")


def extract_feature_array(result: Any) -> np.ndarray:
    payload = result
    if isinstance(payload, list):
        if not payload:
            raise ValueError("FunASR returned an empty list.")
        payload = payload[0]

    if not isinstance(payload, dict):
        raise TypeError(f"Expected FunASR to return a dict-like payload, got {type(payload).__name__}.")

    if "feats" not in payload:
        raise KeyError(f"Expected a `feats` field in the FunASR result. Keys: {sorted(payload.keys())}")

    feature_array = np.asarray(payload["feats"], dtype=np.float32)
    return feature_array


def save_feature_tensor(array: np.ndarray, output_path: Path, granularity: str) -> tuple[int, int]:
    if granularity == "frame":
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D frame-level array, got shape {tuple(array.shape)}.")
        num_frames, feature_dim = int(array.shape[0]), int(array.shape[1])
    else:
        if array.ndim == 1:
            array = np.expand_dims(array, axis=0)
        if array.ndim != 2:
            raise ValueError(f"Expected a 1D or 2D utterance-level array, got shape {tuple(array.shape)}.")
        num_frames, feature_dim = int(array.shape[0]), int(array.shape[1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(array.copy()), output_path)
    return num_frames, feature_dim


def write_manifest(rows: list[dict[str, Any]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "label",
        "agreement",
        "audio_relpath",
        "audio_path",
        "embedding_path",
        "status",
        "num_frames",
        "feature_dim",
        "error",
        "elapsed_seconds",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(Path(__file__).resolve())
    manifest_path = resolve_repo_path(args.manifest_path, repo_root)
    summary_path = resolve_repo_path(args.summary_path, repo_root)

    records = read_extraction_records(args=args, repo_root=repo_root)
    model = init_model(args=args)

    manifest_rows: list[dict[str, Any]] = []
    extracted_count = 0
    skipped_count = 0
    failed_count = 0
    start_time = time.perf_counter()

    print(
        f"Starting extraction with model={args.model_id} hub={args.hub} "
        f"granularity={args.granularity} samples={len(records)} host={socket.gethostname()}"
    )

    for index, record in enumerate(records, start=1):
        row_start = time.perf_counter()
        status = "ok"
        error_message = ""
        num_frames = 0
        feature_dim = 0

        try:
            if not record.audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {record.audio_path}")

            if args.resume and record.output_path.exists():
                tensor = torch.load(record.output_path, map_location="cpu")
                if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
                    raise ValueError(f"Existing embedding is invalid: {record.output_path}")
                skipped_count += 1
                status = "skipped_existing"
                num_frames = int(tensor.shape[0])
                feature_dim = int(tensor.shape[1])
            else:
                result = run_inference(model=model, audio_path=record.audio_path, args=args)
                feature_array = extract_feature_array(result)
                num_frames, feature_dim = save_feature_tensor(
                    array=feature_array,
                    output_path=record.output_path,
                    granularity=args.granularity,
                )
                extracted_count += 1
        except Exception as exc:
            failed_count += 1
            status = "error"
            error_message = f"{type(exc).__name__}: {exc}"

        manifest_rows.append(
            {
                "sample_id": record.sample_id,
                "label": record.label,
                "agreement": record.agreement,
                "audio_relpath": record.audio_relpath,
                "audio_path": str(record.audio_path),
                "embedding_path": str(record.output_path),
                "status": status,
                "num_frames": num_frames,
                "feature_dim": feature_dim,
                "error": error_message,
                "elapsed_seconds": round(time.perf_counter() - row_start, 4),
            }
        )

        if index == 1 or index % 100 == 0 or index == len(records):
            print(
                f"[extract] {index}/{len(records)} status={status} "
                f"ok={extracted_count} skipped={skipped_count} failed={failed_count}"
            )

    write_manifest(rows=manifest_rows, manifest_path=manifest_path)

    summary = {
        "model_id": args.model_id,
        "hub": args.hub,
        "granularity": args.granularity,
        "dataset_csv": resolve_repo_path(args.dataset_csv, repo_root),
        "audio_root": resolve_repo_path(args.audio_root, repo_root),
        "output_root": resolve_repo_path(args.output_root, repo_root),
        "manifest_path": manifest_path,
        "sample_count": len(records),
        "extracted_count": extracted_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "elapsed_seconds": round(time.perf_counter() - start_time, 3),
        "host": socket.gethostname(),
        "resume": args.resume,
        "filters": {
            "include_emotions": args.include_emotions,
            "exclude_emotions": args.exclude_emotions,
            "min_agreement": args.min_agreement,
            "max_samples": args.max_samples,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(to_builtin(summary), handle, indent=2)

    print(f"Manifest saved to: {manifest_path}")
    print(f"Summary saved to: {summary_path}")
    if failed_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
