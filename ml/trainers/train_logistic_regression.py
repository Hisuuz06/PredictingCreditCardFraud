from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

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
    model.fit(X_train, y_train)
    return model.predict_proba(X_validation)[:, 1]


def _save_model(model: Pipeline, path: Path) -> None:
    joblib.dump(model, path)


def train_logistic_regression() -> None:
    train_cross_validation(
        model_name="logistic_regression",
        model_extension="joblib",
        model_factory=_build_model,
        fit_predict=_fit_predict,
        save_model=_save_model,
    )


if __name__ == "__main__":
    train_logistic_regression()
