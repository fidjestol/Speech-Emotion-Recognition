from __future__ import annotations

EMOTION_LABELS: tuple[str, ...] = ("sur", "ang", "hap", "sad", "neu", "fru")
LABEL_TO_ID: dict[str, int] = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
ID_TO_LABEL: dict[int, str] = {idx: label for label, idx in LABEL_TO_ID.items()}

