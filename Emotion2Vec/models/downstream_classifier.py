from __future__ import annotations

import torch
from torch import Tensor, nn


class Emotion2VecDownstreamClassifier(nn.Module):
    """Mean-pool frozen emotion2vec frame embeddings, then classify emotion."""

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 6,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output_proj = nn.Linear(hidden_dim, num_classes)

    def mean_pool(self, embeddings: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Pool valid frames only when a padding mask is provided."""
        if embeddings.ndim != 3:
            raise ValueError(
                f"Expected embeddings with shape (batch, frames, dim), got {tuple(embeddings.shape)}."
            )
        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, got {embeddings.shape[-1]}."
            )

        if attention_mask is None:
            return embeddings.mean(dim=1)

        if attention_mask.ndim != 2:
            raise ValueError(
                f"Expected attention_mask with shape (batch, frames), got {tuple(attention_mask.shape)}."
            )
        if attention_mask.shape != embeddings.shape[:2]:
            raise ValueError(
                "attention_mask must match the first two embedding dimensions "
                f"{tuple(embeddings.shape[:2])}, got {tuple(attention_mask.shape)}."
            )

        frame_mask = attention_mask.to(dtype=embeddings.dtype).unsqueeze(-1)
        pooled_sum = (embeddings * frame_mask).sum(dim=1)
        frame_count = frame_mask.sum(dim=1).clamp_min(1.0)
        return pooled_sum / frame_count

    def forward(self, embeddings: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        pooled = self.mean_pool(embeddings=embeddings, attention_mask=attention_mask)
        hidden = self.activation(self.input_proj(pooled))
        logits = self.output_proj(hidden)
        return logits

