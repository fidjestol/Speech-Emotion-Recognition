"""Self-supervised embedding extraction (HuBERT / wav2vec2)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from feature_extraction.common import (
    configure_cpu_math_threads,
    machine_name_from_env,
    resolve_thread_workers,
    resolve_torch_device,
)
from utils.threading import threaded_map

_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def _require_torch_stack() -> tuple[Any, Any]:
    """Import torch + transformers with a clear error message if missing."""

    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        raise ImportError(
            "SSL embeddings require torch and transformers. "
            "Install with: uv add torch transformers"
        ) from exc
    return torch, (AutoModel, AutoProcessor)


def _get_model(model_name: str, device: str | None) -> tuple[Any, Any, str]:
    """Load or reuse the processor + model for the given name."""

    torch, (AutoModel, AutoProcessor) = _require_torch_stack()
    machine_name = machine_name_from_env()

    if device is None:
        device = resolve_torch_device(machine_name)

    cached = _MODEL_CACHE.get(model_name)
    if cached is None:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        _MODEL_CACHE[model_name] = (processor, model)
    else:
        processor, model = cached

    model = model.to(device)
    return processor, model, device


def _resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target_sr, preferring torchaudio if available."""

    if sr == target_sr:
        return audio.astype(np.float32, copy=False)

    try:  # pragma: no cover - optional
        import torch
        import torchaudio

        waveform = torch.from_numpy(audio.astype(np.float32, copy=False))
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=target_sr,
        )
        resampled = resampler(waveform).squeeze(0).cpu().numpy()
        return resampled.astype(np.float32, copy=False)
    except Exception:
        import librosa

        return librosa.resample(
            audio.astype(np.float32, copy=False),
            orig_sr=sr,
            target_sr=target_sr,
        ).astype(np.float32, copy=False)


def extract(
    audio: np.ndarray,
    sr: int,
    *,
    model_name: str = "facebook/hubert-base-ls960",
    device: str | None = None,
    target_sr: int = 16000,
    pool: tuple[str, ...] = ("mean", "std"),
) -> dict[str, np.ndarray]:
    """Extract pooled SSL embeddings from audio."""

    torch, _ = _require_torch_stack()
    processor, model, device = _get_model(model_name, device)

    audio = np.asarray(audio, dtype=np.float32)
    audio = _resample(audio, sr, target_sr)

    inputs = processor(audio, sampling_rate=target_sr, return_tensors="pt")
    input_values = inputs["input_values"].to(device)

    with torch.no_grad():
        outputs = model(input_values)
        hidden = outputs.last_hidden_state[0]

    frames, dim = hidden.shape
    features: dict[str, np.ndarray] = {
        "ssl_frames": np.asarray(frames, dtype=np.int64),
        "ssl_dim": np.asarray(dim, dtype=np.int64),
    }

    if "mean" in pool:
        mean_vec = hidden.mean(dim=0).float().cpu().numpy().astype(np.float32, copy=False)
        features["ssl_mean"] = mean_vec
    if "std" in pool:
        std_vec = hidden.std(dim=0, unbiased=False).float().cpu().numpy().astype(
            np.float32,
            copy=False,
        )
        features["ssl_std"] = std_vec

    return features


def extract_batch(
    items: list[tuple[np.ndarray, int]],
    *,
    max_workers: int | None = None,
    use_threads_if_available: bool = True,
    model_name: str = "facebook/hubert-base-ls960",
    device: str | None = None,
    target_sr: int = 16000,
    pool: tuple[str, ...] = ("mean", "std"),
) -> list[dict[str, np.ndarray]]:
    """Extract pooled SSL embeddings for many utterances."""

    if max_workers is None and machine_name_from_env() == "macbook":
        max_workers = resolve_thread_workers("macbook")
        configure_cpu_math_threads("macbook", num_threads=max(1, min(max_workers, os.cpu_count() or 1)))

    def _worker(item: tuple[np.ndarray, int]) -> dict[str, np.ndarray]:
        audio, sr = item
        return extract(
            audio,
            sr,
            model_name=model_name,
            device=device,
            target_sr=target_sr,
            pool=pool,
        )

    return threaded_map(
        items,
        _worker,
        max_workers=max_workers,
        use_threads_if_available=use_threads_if_available,
    )
