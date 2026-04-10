from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_selection.common import DEFAULT_EXCLUDED_EMOTIONS, DEFAULT_REQUIRE_AGREEMENT
from feature_family_selection.genetic_algorithm.common import TARGET_COLUMN, merge_selected_families


@dataclass(slots=True)
class FamilySubsetEvaluation:
    fitness: float
    accuracy: float
    f1_macro: float
    f1_weighted: float
    num_selected_families: int
    num_features: int
    num_rows: int
    num_train_rows: int
    num_test_rows: int
    selected_families: list[str]

    def to_dict(self) -> dict[str, float | int | list[str]]:
        return {
            "fitness": self.fitness,
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "num_selected_families": self.num_selected_families,
            "num_features": self.num_features,
            "num_rows": self.num_rows,
            "num_train_rows": self.num_train_rows,
            "num_test_rows": self.num_test_rows,
            "selected_families": self.selected_families,
        }


def build_evaluator(model_name: str, random_state: int):
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "logreg":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        solver="lbfgs",
                        multi_class="auto",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_name == "linear_svc":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LinearSVC(random_state=random_state, dual="auto")),
            ]
        )
    raise ValueError(f"Unsupported evaluator model: {model_name}")


def evaluate_family_subset(
    *,
    repo_root: Path,
    family_keys: Sequence[str],
    include_xxx: bool = False,
    require_agreement: bool = DEFAULT_REQUIRE_AGREEMENT,
    excluded_emotions: Sequence[str] = DEFAULT_EXCLUDED_EMOTIONS,
    evaluator_model: str = "random_forest",
    test_size: float = 0.20,
    random_state: int = 42,
    alpha: float = 1.0,
    beta: float = 0.05,
    family_space_size: int | None = None,
) -> FamilySubsetEvaluation:
    merged = merge_selected_families(
        repo_root,
        family_keys,
        include_xxx=include_xxx,
        require_agreement=require_agreement,
        excluded_emotions=excluded_emotions,
    )
    X = merged.drop(columns=[TARGET_COLUMN])
    y = merged[TARGET_COLUMN].astype(str)

    metadata_columns = [col for col in X.columns if "__" not in col]
    X = X.drop(columns=metadata_columns)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    evaluator = build_evaluator(evaluator_model, random_state=random_state)
    evaluator.fit(X_train, y_train)
    y_pred = evaluator.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    family_space_size = family_space_size or len(family_keys)
    penalty = beta * (len(family_keys) / max(1, family_space_size))
    fitness = alpha * f1_macro - penalty

    return FamilySubsetEvaluation(
        fitness=float(fitness),
        accuracy=float(accuracy),
        f1_macro=float(f1_macro),
        f1_weighted=float(f1_weighted),
        num_selected_families=len(family_keys),
        num_features=int(X.shape[1]),
        num_rows=int(X.shape[0]),
        num_train_rows=int(X_train.shape[0]),
        num_test_rows=int(X_test.shape[0]),
        selected_families=list(family_keys),
    )
