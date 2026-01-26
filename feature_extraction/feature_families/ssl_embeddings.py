"""Self-supervised embedding extraction (HuBERT / wav2vec2)."""

from __future__ import annotations

from typing import Any

import numpy as np

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

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
