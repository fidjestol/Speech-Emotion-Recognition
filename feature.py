from pathlib import Path

from datasets import Audio, load_dataset

root = Path(__file__).resolve().parent
cache_dir = root / "datasets" / "IEMOCAP"
cache_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("AbstractTTS/IEMOCAP", cache_dir=str(cache_dir))

# Avoid decoding audio until the environment has torchcodec/FFmpeg set up.
audio_cols = [name for name, feat in dataset["train"].features.items() if isinstance(feat, Audio)]
for col in audio_cols:
    dataset = dataset.cast_column(col, Audio(decode=False))

print(dataset["train"][0])
