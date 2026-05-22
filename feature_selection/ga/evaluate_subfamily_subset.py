from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_family_selection.genetic_algorithm.evaluate_family_subset import build_evaluator
from feature_family_selection.genetic_algorithm.common import merge_selected_families
from feature_selection.common import DEFAULT_EXCLUDED_EMOTIONS, DEFAULT_REQUIRE_AGREEMENT

from feature_selection.ga.common import TARGET_COLUMN
from feature_selection.ga.subfamily_groups import SubfamilyGroup


@dataclass(slots=True)
class SubfamilySubsetEvaluation:
    fitness: float
    accuracy: float
    f1_macro: float
    f1_weighted: float
    num_selected_groups: int
    num_available_groups: int
    num_selected_features: int
    num_available_features: int
    num_rows: int
    num_train_rows: int
    num_test_rows: int
    validation_mode: str
    n_folds: int
    selected_groups: list[str]
    selected_families: list[str]

    def to_dict(self) -> dict[str, float | int | list[str]]:
        return {
            "fitness": self.fitness,
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "num_selected_groups": self.num_selected_groups,
            "num_available_groups": self.num_available_groups,
            "num_selected_features": self.num_selected_features,
            "num_available_features": self.num_available_features,
            "num_rows": self.num_rows,
            "num_train_rows": self.num_train_rows,
            "num_test_rows": self.num_test_rows,
            "validation_mode": self.validation_mode,
            "n_folds": self.n_folds,
            "selected_groups": self.selected_groups,
            "selected_families": self.selected_families,
        }


def selected_columns_from_groups(groups: Sequence[SubfamilyGroup], selected_group_keys: Sequence[str]) -> list[str]:
    selected = set(selected_group_keys)
    columns: list[str] = []
    for group in groups:
        if group.key in selected:
            columns.extend(group.columns)
    return sorted(set(columns))


def evaluate_subfamily_subset(
    *,
    repo_root: Path,
    groups: Sequence[SubfamilyGroup],
    selected_group_keys: Sequence[str],
    include_xxx: bool = False,
    require_agreement: bool = DEFAULT_REQUIRE_AGREEMENT,
    excluded_emotions: Sequence[str] = DEFAULT_EXCLUDED_EMOTIONS,
    evaluator_model: str = "random_forest",
    test_size: float = 0.20,
    random_state: int = 42,
    alpha: float = 1.0,
    beta: float = 0.03,
    gamma: float = 0.02,
    validation_mode: str = "stratified",
    group_column: str = "session",
) -> SubfamilySubsetEvaluation:
    if not selected_group_keys:
        raise ValueError("At least one sub-family group must be selected.")

    selected_groups = [group for group in groups if group.key in set(selected_group_keys)]
    selected_families = sorted({group.family for group in selected_groups})
    selected_columns = selected_columns_from_groups(groups, selected_group_keys)
    all_columns = sorted({column for group in groups for column in group.columns})
    if not selected_columns:
        raise ValueError("Selected sub-family groups did not resolve to any feature columns.")

    merged = merge_selected_families(
        repo_root,
        selected_families,
        include_xxx=include_xxx,
        require_agreement=require_agreement,
        excluded_emotions=excluded_emotions,
    )
    missing_columns = [column for column in selected_columns if column not in merged.columns]
    if missing_columns:
        preview = ", ".join(missing_columns[:10])
        raise KeyError(f"{len(missing_columns)} selected columns were missing after merge: {preview}")

    X = merged[selected_columns].copy()
    y = merged[TARGET_COLUMN].astype(str)
    groups_by_row = merged[group_column] if group_column in merged.columns else None
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if validation_mode == "stratified":
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
        num_train_rows = int(X_train.shape[0])
        num_test_rows = int(X_test.shape[0])
        n_folds = 1
    elif validation_mode == "loso":
        if groups_by_row is None:
            raise KeyError(f"LOSO validation requires group column '{group_column}'.")
        logo = LeaveOneGroupOut()
        fold_rows: list[dict[str, float]] = []
        train_sizes: list[int] = []
        test_sizes: list[int] = []
        for fold_index, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups_by_row), start=1):
            evaluator = build_evaluator(evaluator_model, random_state=random_state + fold_index)
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            evaluator.fit(X_train, y_train)
            y_pred = evaluator.predict(X_test)
            fold_rows.append(
                {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                    "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                }
            )
            train_sizes.append(int(X_train.shape[0]))
            test_sizes.append(int(X_test.shape[0]))
        accuracy = float(np.mean([row["accuracy"] for row in fold_rows]))
        f1_macro = float(np.mean([row["f1_macro"] for row in fold_rows]))
        f1_weighted = float(np.mean([row["f1_weighted"] for row in fold_rows]))
        num_train_rows = int(round(float(np.mean(train_sizes))))
        num_test_rows = int(round(float(np.mean(test_sizes))))
        n_folds = len(fold_rows)
    else:
        raise ValueError("validation_mode must be one of: stratified, loso")

    group_penalty = beta * (len(selected_group_keys) / max(1, len(groups)))
    feature_penalty = gamma * (len(selected_columns) / max(1, len(all_columns)))
    fitness = alpha * f1_macro - group_penalty - feature_penalty

    return SubfamilySubsetEvaluation(
        fitness=float(fitness),
        accuracy=float(accuracy),
        f1_macro=float(f1_macro),
        f1_weighted=float(f1_weighted),
        num_selected_groups=len(selected_group_keys),
        num_available_groups=len(groups),
        num_selected_features=len(selected_columns),
        num_available_features=len(all_columns),
        num_rows=int(X.shape[0]),
        num_train_rows=num_train_rows,
        num_test_rows=num_test_rows,
        validation_mode=validation_mode,
        n_folds=n_folds,
        selected_groups=list(selected_group_keys),
        selected_families=selected_families,
    )
