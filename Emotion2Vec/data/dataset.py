from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from Emotion2Vec import EMOTION_LABELS, LABEL_TO_ID


@dataclass(frozen=True)
class Emotion2VecRecord:
    sample_id: str
    label: str
    label_id: int
    embedding_path: Path


class Emotion2VecEmbeddingDataset(Dataset[dict[str, Any]]):
    """Load pre-extracted emotion2vec frame embeddings from .pt files."""

    def __init__(
        self,
        csv_path: str | Path | pd.DataFrame,
        *,
        repo_root: str | Path | None = None,
        label_col: str = "label",
        embedding_path_col: str = "embedding_path",
        audio_path_col: str | None = None,
        sample_id_col: str | None = None,
        embeddings_root: str | Path | None = None,
        embedding_suffix: str = ".pt",
        expected_dim: int = 768,
        allowed_labels: tuple[str, ...] = EMOTION_LABELS,
    ) -> None:
        if isinstance(csv_path, pd.DataFrame):
            self.csv_path = None
            frame = csv_path.copy()
        else:
            self.csv_path = Path(csv_path).expanduser().resolve()
            if not self.csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            frame = pd.read_csv(self.csv_path)

        self.repo_root = (
            Path(repo_root).expanduser().resolve()
            if repo_root is not None
            else Path.cwd().resolve()
        )
        self.label_col = label_col
        self.embedding_path_col = embedding_path_col
        self.audio_path_col = audio_path_col
        self.sample_id_col = sample_id_col
        self.embeddings_root = (
            Path(embeddings_root).expanduser()
            if embeddings_root is not None
            else None
        )
        self.embedding_suffix = embedding_suffix
        self.expected_dim = expected_dim
        self.allowed_labels = allowed_labels

        required_cols = {label_col}
        if embedding_path_col not in frame.columns:
            if audio_path_col is None or audio_path_col not in frame.columns:
                raise ValueError(
                    f"CSV must contain '{embedding_path_col}' or '{audio_path_col}' to locate embeddings."
                )
            if self.embeddings_root is None:
                raise ValueError(
                    "embeddings_root is required when deriving .pt paths from an audio path column."
                )
        else:
            required_cols.add(embedding_path_col)

        missing_cols = sorted(col for col in required_cols if col not in frame.columns)
        if missing_cols:
            raise ValueError(f"Missing required CSV column(s): {', '.join(missing_cols)}")

        self.records: list[Emotion2VecRecord] = []
        invalid_labels: list[str] = []

        for row_idx, row in frame.iterrows():
            label = str(row[label_col]).strip().lower()
            if label not in LABEL_TO_ID or label not in allowed_labels:
                invalid_labels.append(label)
                continue

            embedding_path = self._resolve_embedding_path(row=row)
            sample_id = self._resolve_sample_id(row=row, row_idx=row_idx, embedding_path=embedding_path)

            self.records.append(
                Emotion2VecRecord(
                    sample_id=sample_id,
                    label=label,
                    label_id=LABEL_TO_ID[label],
                    embedding_path=embedding_path,
                )
            )

        if invalid_labels:
            unique_invalid = ", ".join(sorted(set(invalid_labels)))
            raise ValueError(
                "Found labels outside the 6-way downstream setup. "
                f"Unexpected labels: {unique_invalid}. "
                "Prepare 6-way train/val/test CSVs before training."
            )

        if not self.records:
            source = self.csv_path if self.csv_path is not None else "in-memory DataFrame"
            raise RuntimeError(f"No valid samples were loaded from {source}.")

    def _resolve_embedding_path(self, row: pd.Series) -> Path:
        if self.embedding_path_col in row.index and pd.notna(row[self.embedding_path_col]):
            raw_path = Path(str(row[self.embedding_path_col]).strip())
            return self._resolve_path(raw_path)

        if self.audio_path_col is None or self.audio_path_col not in row.index:
            raise ValueError("No embedding path column available to resolve embeddings.")

        raw_audio_path = Path(str(row[self.audio_path_col]).replace("\\", "/").strip())
        if raw_audio_path.is_absolute():
            try:
                raw_audio_path = raw_audio_path.relative_to(self.repo_root)
            except ValueError:
                raw_audio_path = Path(raw_audio_path.name)
        relative_path = raw_audio_path.with_suffix(self.embedding_suffix)
        embedding_root = self._resolve_path(self.embeddings_root)
        return (embedding_root / relative_path).resolve()

    def _resolve_sample_id(self, row: pd.Series, row_idx: int, embedding_path: Path) -> str:
        if self.sample_id_col is not None and self.sample_id_col in row.index and pd.notna(row[self.sample_id_col]):
            sample_id = str(row[self.sample_id_col]).strip()
            if sample_id:
                return sample_id
        return f"{embedding_path.stem}_{row_idx}"

    def _resolve_path(self, raw_path: Path | None) -> Path:
        if raw_path is None:
            raise ValueError("Expected a valid path value, got None.")
        expanded = raw_path.expanduser()
        if expanded.is_absolute():
            return expanded.resolve()
        return (self.repo_root / expanded).resolve()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        embeddings = self._load_embedding_tensor(record.embedding_path)

        return {
            "embeddings": embeddings,
            "label_id": record.label_id,
            "label_str": record.label,
            "sample_id": record.sample_id,
            "embedding_path": str(record.embedding_path),
            "num_frames": int(embeddings.shape[0]),
        }

    def _load_embedding_tensor(self, embedding_path: Path) -> Tensor:
        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

        payload = torch.load(embedding_path, map_location="cpu")
        tensor: Tensor

        if isinstance(payload, Tensor):
            tensor = payload
        elif isinstance(payload, dict):
            tensor = self._extract_tensor_from_dict(payload=payload, embedding_path=embedding_path)
        else:
            raise TypeError(
                f"Unsupported payload type in {embedding_path}: {type(payload).__name__}."
            )

        tensor = tensor.detach().to(dtype=torch.float32)
        if tensor.ndim != 2:
            raise ValueError(
                f"Expected 2D emotion2vec embeddings in {embedding_path}, got shape {tuple(tensor.shape)}."
            )
        if tensor.shape[0] == 0:
            raise ValueError(f"Embedding tensor in {embedding_path} has zero frames.")
        if tensor.shape[1] != self.expected_dim:
            raise ValueError(
                f"Expected embedding dim {self.expected_dim} in {embedding_path}, got {tensor.shape[1]}."
            )
        return tensor

    @staticmethod
    def _extract_tensor_from_dict(payload: dict[str, Any], embedding_path: Path) -> Tensor:
        for key in ("embeddings", "embedding", "emotion2vec", "features", "hidden_states"):
            value = payload.get(key)
            if isinstance(value, Tensor):
                return value
        raise KeyError(
            f"Expected a tensor under one of [embeddings, embedding, emotion2vec, features, hidden_states] in {embedding_path}."
        )


def collate_emotion2vec_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length frame sequences and build a valid-frame mask."""
    embeddings = [sample["embeddings"] for sample in batch]
    labels = torch.tensor([sample["label_id"] for sample in batch], dtype=torch.long)
    lengths = torch.tensor([sample["num_frames"] for sample in batch], dtype=torch.long)
    padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0.0)

    frame_positions = torch.arange(padded_embeddings.shape[1]).unsqueeze(0)
    attention_mask = frame_positions < lengths.unsqueeze(1)

    return {
        "embeddings": padded_embeddings,
        "attention_mask": attention_mask,
        "labels": labels,
        "lengths": lengths,
        "sample_ids": [sample["sample_id"] for sample in batch],
        "label_strs": [sample["label_str"] for sample in batch],
        "embedding_paths": [sample["embedding_path"] for sample in batch],
    }
