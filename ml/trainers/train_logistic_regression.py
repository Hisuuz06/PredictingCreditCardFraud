from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.estimators.logistic_regression_model import build_model
from ml.trainers.cv_training import train_cross_validation

RANDOM_STATE = 42


def _build_model(
    X_train: pd.DataFrame,
    _y_train: pd.Series,
    categorical_columns: list[str],
) -> Pipeline:
    boolean_columns = X_train.select_dtypes(include=["bool"]).columns.tolist()
    numeric_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
    classified_columns = (
        set(boolean_columns) | set(numeric_columns) | set(categorical_columns)
    )
    unclassified_columns = set(X_train.columns) - classified_columns
    if unclassified_columns:
        raise ValueError(
            f"Unsupported feature types: {sorted(unclassified_columns)}"
        )

    return build_model(
        numeric_columns,
        categorical_columns,
        boolean_columns,
        random_state=RANDOM_STATE,
    )


def _fit_predict(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    _categorical_columns: list[str],
) -> np.ndarray:
    # SimpleImputer in current scikit-learn versions does not accept bool dtype.
    # Convert boolean features to 0/1 while preserving the shared feature set.
    boolean_columns = X_train.select_dtypes(include=["bool"]).columns
    if len(boolean_columns):
        X_train = X_train.copy()
        X_validation = X_validation.copy()
        X_train[boolean_columns] = X_train[boolean_columns].astype("int8")
        X_validation[boolean_columns] = X_validation[boolean_columns].astype("int8")

    model.fit(X_train, y_train)
    return model.predict_proba(X_validation)[:, 1]


def _save_model(model: Pipeline, path: Path) -> None:
    joblib.dump(model, path)


def train_logistic_regression(
    feature_columns: list[str] | None = None, *, feature_version: str | None = None
) -> None:
    if feature_columns is None:
        columns = pd.read_csv(ROOT / "data" / "raw" / "train.csv", nrows=0).columns
        feature_columns = [column for column in columns if column not in {"id", "is_fraud"}]
    train_cross_validation(
        model_name="logistic_regression",
        model_extension="joblib",
        feature_columns=feature_columns,
        model_factory=_build_model,
        fit_predict=_fit_predict,
        save_model=_save_model,
        feature_version=feature_version,
    )


if __name__ == "__main__":
    train_logistic_regression()
