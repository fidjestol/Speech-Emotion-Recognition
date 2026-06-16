from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class SERModelPackage:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.manifest = self._read_json(self.model_dir / "manifest.json")
        self.schema = self._read_json(self.model_dir / "feature_schema.json")
        self.model = joblib.load(self.model_dir / self.manifest["model_file"])

    @staticmethod
    def _read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def required_features(self) -> list[str]:
        return list(self.schema.get("numeric_features", []))

    @property
    def requires_text(self) -> bool:
        return bool(self.schema.get("requires_text", False))

    def build_frame(self, payload: dict[str, Any], *, strict: bool = True) -> pd.DataFrame:
        row: dict[str, Any] = {}
        if self.requires_text:
            text = payload.get("text")
            if strict and (text is None or not str(text).strip()):
                raise ValueError(f"{self.manifest['model_id']} requires a non-empty 'text' field.")
            row["text"] = "" if text is None else str(text)

        feature_payload = payload.get("features") or {}
        if not isinstance(feature_payload, dict):
            raise TypeError("'features' must be an object mapping feature names to numeric values.")

        missing = [name for name in self.required_features if name not in feature_payload]
        if strict and missing:
            sample = ", ".join(missing[:8])
            suffix = "" if len(missing) <= 8 else f" ... ({len(missing)} missing total)"
            raise ValueError(f"{self.manifest['model_id']} missing numeric features: {sample}{suffix}")

        for name in self.required_features:
            row[name] = feature_payload.get(name)
        return pd.DataFrame([row])

    def predict(self, payload: dict[str, Any], *, strict: bool = True) -> dict[str, Any]:
        frame = self.build_frame(payload, strict=strict)
        label = str(self.model.predict(frame)[0])
        result: dict[str, Any] = {
            "model_id": self.manifest["model_id"],
            "modality": self.manifest["modality"],
            "tier": self.manifest["tier"],
            "emotion": label,
        }
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(frame)[0]
            result["probabilities"] = {
                str(label_name): float(prob)
                for label_name, prob in zip(self.model.classes_, probabilities, strict=True)
            }
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single prediction from an exported SER model package.")
    parser.add_argument("model_dir", type=Path, help="Path to one model package directory.")
    parser.add_argument(
        "--payload",
        required=True,
        help="JSON payload with optional 'text' and 'features' fields.",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Allow missing numeric features; the model imputer will fill them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package = SERModelPackage(args.model_dir)
    payload = json.loads(args.payload)
    print(json.dumps(package.predict(payload, strict=not args.non_strict), indent=2))


if __name__ == "__main__":
    main()
