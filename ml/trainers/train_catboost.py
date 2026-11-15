from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from ml.estimators.catboost_model import build_model
from ml.trainers.cv_training import train_cross_validation


def _fit_predict(
    model: CatBoostClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    categorical_columns: list[str],
) -> np.ndarray:
    if X_train[categorical_columns].isna().any().any():
        raise ValueError("CatBoost categorical features must not contain missing values.")
    model.fit(X_train, y_train, cat_features=categorical_columns)
    return model.predict_proba(X_validation)[:, 1]


def _save_model(model: CatBoostClassifier, path: Path) -> None:
    model.save_model(str(path))


def train_catboost() -> None:
    train_cross_validation(
        model_name="catboost",
        model_extension="cbm",
        model_factory=lambda _X_train, _y_train, _categorical_columns: build_model(),
        fit_predict=_fit_predict,
        save_model=_save_model,
    )


if __name__ == "__main__":
    train_catboost()
