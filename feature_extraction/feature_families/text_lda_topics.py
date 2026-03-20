from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def _bootstrap_repo_path() -> Path:
    for path in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if path.is_dir() and (path / "pyproject.toml").exists():
            repo_root = path
            break
    else:
        raise FileNotFoundError("Could not locate repository root from script path.")

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


_REPO_ROOT = _bootstrap_repo_path()

from feature_extraction.common import configure_cpu_math_threads, find_repo_root, machine_name_from_env


METADATA_COLUMNS = [
    "path",
    "session",
    "method",
    "gender",
    "emotion",
    "n_annotators",
    "agreement",
    "utt_id",
    "text",
    "split",
]

BRACKETED_STAGE_RE = re.compile(r"\[[^\]]+\]")
MULTISPACE_RE = re.compile(r"\s+")
UTT_ORDER_RE = re.compile(r"_(\d+)$")


@dataclass(frozen=True)
class LDAExtractionConfig:
    source_csv: Path
    output_csv: Path
    metadata_json: Path
    top_words_csv: Path
    n_topics: int = 32
    max_features: int = 5000
    min_df: int = 3
    max_df: float = 0.85
    max_iter: int = 25
    learning_method: str = "batch"
    random_state: int = 42
    context_window: int = 1
    vectorizer_stop_words: str = "english"
    topic_prefix: str = "lda_topic"
    source_text_column: str = "text"


def _read_source_frame(source_csv: Path) -> pd.DataFrame:
    if not source_csv.exists():
        raise FileNotFoundError(f"Missing source CSV: {source_csv}")

    header = pd.read_csv(source_csv, nrows=0).columns.tolist()
    missing = [col for col in METADATA_COLUMNS if col not in header]
    if missing:
        raise ValueError(f"Source CSV is missing required columns: {missing}")

    df = pd.read_csv(source_csv, usecols=METADATA_COLUMNS)
    df["path"] = df["path"].astype(str).str.replace("\\", "/", regex=False).str.strip()
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
    df["method"] = df["method"].astype(str).str.strip().str.lower()
    df["gender"] = df["gender"].astype(str).str.strip().str.upper()
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["utt_id"] = df["utt_id"].astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str)
    return df


def ensure_train_test_split(df: pd.DataFrame, *, test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    work = df.copy()
    split_counts = work["split"].value_counts(dropna=False).to_dict()
    has_train = "train" in split_counts
    has_test = "test" in split_counts
    if has_train and has_test:
        return work

    keep_idx = work.index.to_numpy()
    train_idx, test_idx = train_test_split(
        keep_idx,
        test_size=test_size,
        random_state=random_state,
        stratify=work["emotion"].astype(str),
    )

    work["split"] = "train"
    work.loc[test_idx, "split"] = "test"
    return work


def normalize_text(text: str) -> str:
    text = BRACKETED_STAGE_RE.sub(" ", str(text))
    text = text.replace("_", " ").replace("-", " ")
    text = MULTISPACE_RE.sub(" ", text).strip().lower()
    return text


def _dialogue_id_from_path(path_str: str) -> str:
    path = Path(path_str)
    return path.parent.name


def _utt_order_from_utt_id(utt_id: str) -> int:
    match = UTT_ORDER_RE.search(str(utt_id))
    if match:
        return int(match.group(1))
    return 0


def build_dialogue_context_frame(source_df: pd.DataFrame, context_window: int) -> pd.DataFrame:
    work = source_df.copy()
    work["_dialogue_id"] = work["path"].map(_dialogue_id_from_path)
    work["_utt_order"] = work["utt_id"].map(_utt_order_from_utt_id)
    work = work.sort_values(["session", "_dialogue_id", "_utt_order", "utt_id"]).reset_index(drop=True)

    context_texts: list[str] = [""] * len(work)
    grouped = work.groupby(["session", "_dialogue_id"], sort=False)
    for _, idx in grouped.groups.items():
        group_positions = list(idx)
        texts = work.loc[group_positions, "text"].tolist()
        for local_i, frame_i in enumerate(group_positions):
            start = max(0, local_i - context_window)
            end = min(len(texts), local_i + context_window + 1)
            joined = " ".join(str(t) for t in texts[start:end] if str(t).strip())
            context_texts[frame_i] = joined

    out = work[METADATA_COLUMNS].copy()
    out["text"] = context_texts
    return out


def _fit_vectorizer_and_lda(
    texts_train: pd.Series,
    cfg: LDAExtractionConfig,
) -> tuple[CountVectorizer, LatentDirichletAllocation]:
    vectorizer = CountVectorizer(
        lowercase=False,
        max_features=cfg.max_features,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        stop_words=cfg.vectorizer_stop_words,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
    )
    x_train = vectorizer.fit_transform(texts_train)
    if x_train.shape[1] == 0:
        raise ValueError("CountVectorizer produced zero features. Relax min_df/max_df or inspect text coverage.")

    lda = LatentDirichletAllocation(
        n_components=cfg.n_topics,
        max_iter=cfg.max_iter,
        learning_method=cfg.learning_method,
        random_state=cfg.random_state,
    )
    lda.fit(x_train)
    return vectorizer, lda


def topic_feature_names(prefix: str, n_topics: int) -> list[str]:
    return [f"{prefix}_{idx:03d}" for idx in range(n_topics)]


def _config_to_jsonable(cfg: LDAExtractionConfig) -> dict[str, Any]:
    raw = asdict(cfg)
    return {key: str(value) if isinstance(value, Path) else value for key, value in raw.items()}


def transform_with_lda(
    source_df: pd.DataFrame,
    cfg: LDAExtractionConfig,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    work = source_df.copy()
    work["text"] = work[cfg.source_text_column].map(normalize_text)

    usable_mask = work["text"].str.len() > 0
    split_mask = work["split"].eq("train")
    train_mask = usable_mask & split_mask

    if not train_mask.any():
        raise ValueError("No non-empty training texts found. Expected split == 'train' rows with text.")

    vectorizer, lda = _fit_vectorizer_and_lda(work.loc[train_mask, "text"], cfg)
    x_all = vectorizer.transform(work["text"])
    topic_matrix = lda.transform(x_all)

    feature_names = topic_feature_names(cfg.topic_prefix, cfg.n_topics)
    features_df = pd.DataFrame(topic_matrix, columns=feature_names, index=work.index)
    out_df = pd.concat([work[METADATA_COLUMNS].copy(), features_df], axis=1)

    vocab = vectorizer.get_feature_names_out()
    top_word_rows: list[dict[str, Any]] = []
    for topic_idx, weights in enumerate(lda.components_):
        top_indices = weights.argsort()[::-1][:15]
        for rank, word_idx in enumerate(top_indices, start=1):
            top_word_rows.append(
                {
                    "topic_index": int(topic_idx),
                    "rank": int(rank),
                    "word": str(vocab[word_idx]),
                    "weight": float(weights[word_idx]),
                }
            )
    top_words_df = pd.DataFrame(top_word_rows)

    metadata = {
        "source_csv": str(cfg.source_csv),
        "output_csv": str(cfg.output_csv),
        "n_rows": int(len(work)),
        "n_train_rows": int(train_mask.sum()),
        "n_test_rows": int(work["split"].eq("test").sum()),
        "n_topics": int(cfg.n_topics),
        "max_features": int(cfg.max_features),
        "min_df": int(cfg.min_df),
        "max_df": float(cfg.max_df),
        "max_iter": int(cfg.max_iter),
        "learning_method": cfg.learning_method,
        "random_state": int(cfg.random_state),
        "vectorizer_stop_words": cfg.vectorizer_stop_words,
        "topic_prefix": cfg.topic_prefix,
        "vocabulary_size": int(len(vocab)),
        "empty_text_rows": int((~usable_mask).sum()),
        "config": _config_to_jsonable(cfg),
    }

    return out_df, metadata, top_words_df


def export_lda_features(source_df: pd.DataFrame, cfg: LDAExtractionConfig) -> dict[str, Any]:
    out_df, metadata, top_words_df = transform_with_lda(source_df, cfg)

    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(cfg.output_csv, index=False)
    cfg.metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    top_words_df.to_csv(cfg.top_words_csv, index=False)
    return metadata


def extract_lda_topics_corpus(
    repo_root: Path,
    *,
    n_topics: int = 32,
    max_features: int = 5000,
    min_df: int = 3,
    max_df: float = 0.85,
    max_iter: int = 25,
    context_window: int = 1,
    random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    source_csv = repo_root / "extracted_features" / "text" / "tfidf_features.csv"
    out_dir = repo_root / "extracted_features" / "text"

    source_df = ensure_train_test_split(
        _read_source_frame(source_csv),
        random_state=random_state,
    )

    utterance_cfg = LDAExtractionConfig(
        source_csv=source_csv,
        output_csv=out_dir / "lda_topics.csv",
        metadata_json=out_dir / "lda_topics_metadata.json",
        top_words_csv=out_dir / "lda_topics_top_words.csv",
        n_topics=n_topics,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        max_iter=max_iter,
        context_window=context_window,
        random_state=random_state,
        topic_prefix="lda_topic",
    )

    context_cfg = LDAExtractionConfig(
        source_csv=source_csv,
        output_csv=out_dir / "lda_topics_context.csv",
        metadata_json=out_dir / "lda_topics_context_metadata.json",
        top_words_csv=out_dir / "lda_topics_context_top_words.csv",
        n_topics=n_topics,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        max_iter=max_iter,
        context_window=context_window,
        random_state=random_state,
        topic_prefix="lda_ctx_topic",
    )

    utterance_meta = export_lda_features(source_df, utterance_cfg)
    context_source_df = build_dialogue_context_frame(source_df, context_window=context_window)
    context_meta = export_lda_features(context_source_df, context_cfg)

    return {
        "lda_topics": utterance_meta,
        "lda_topics_context": context_meta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract utterance-level and dialogue-context LDA topic features.")
    parser.add_argument("--repo-root", type=Path, default=None, help="Repository root (auto-detected by default)")
    parser.add_argument("--n-topics", type=int, default=32)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--max-df", type=float, default=0.85)
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--context-window", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve() if args.repo_root is not None else find_repo_root(Path.cwd().resolve())

    machine_name = machine_name_from_env()
    configure_cpu_math_threads(machine_name)

    results = extract_lda_topics_corpus(
        repo_root,
        n_topics=args.n_topics,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        max_iter=args.max_iter,
        context_window=args.context_window,
        random_state=args.random_state,
    )

    for family, metadata in results.items():
        print(
            f"[{family}] rows={metadata['n_rows']} train_rows={metadata['n_train_rows']} "
            f"topics={metadata['n_topics']} vocab={metadata['vocabulary_size']} "
            f"output={metadata['output_csv']}"
        )


if __name__ == "__main__":
    main()
