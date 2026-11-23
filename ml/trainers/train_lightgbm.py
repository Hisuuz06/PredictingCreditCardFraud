from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.trainers.cv_training import train_cross_validation
from ml.estimators.lightgbm_model import build_model


def _category_frames(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = X_train.copy()
    X_validation = X_validation.copy()
    for column in categorical_columns:
        X_train[column] = X_train[column].astype("category")
        categories = X_train[column].cat.categories
        X_validation[column] = pd.Categorical(
            X_validation[column],
            categories=categories,
        )
    return X_train, X_validation


def _fit_predict(
    model: LGBMClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    categorical_columns: list[str],
) -> np.ndarray:
    X_train, X_validation = _category_frames(
        X_train,
        X_validation,
        categorical_columns,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_validation)[:, 1]


def _save_model(model: LGBMClassifier, path: Path) -> None:
    model.booster_.save_model(str(path))


def train_lightgbm(
    feature_columns: list[str] | None = None, *, feature_version: str | None = None
) -> None:
    if feature_columns is None:
        columns = pd.read_csv(ROOT / "data" / "raw" / "train.csv", nrows=0).columns
        feature_columns = [column for column in columns if column not in {"id", "is_fraud"}]
    train_cross_validation(
        model_name="lightgbm",
        model_extension="txt",
        feature_columns=feature_columns,
        model_factory=lambda _X_train, _y_train, _categorical_columns: build_model(),
        fit_predict=_fit_predict,
        save_model=_save_model,
        feature_version=feature_version,
    )


if __name__ == "__main__":
    train_lightgbm()
