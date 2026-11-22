"""Predict with models trained by train_ml_feature_versions.

Run from the project root:
    python -m ml.scripts.predict_ml_feature_versions --version v1
    python -m ml.scripts.predict_ml_feature_versions --version v2
    python -m ml.scripts.predict_ml_feature_versions --version all

Predictions average the five fold models and are saved separately for each
feature version under predictions/test/ml_vN and submissions/ml_vN.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
TEST_PATH = ROOT / "data" / "raw" / "test.csv"
SAMPLE_PATH = ROOT / "data" / "raw" / "sample_submission.csv"
N_FOLDS = 5
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    if list(sample.columns) != ["id", "is_fraud"]:
        raise ValueError("sample_submission.csv must contain id,is_fraud columns.")
    if not test["id"].is_unique or not sample["id"].is_unique:
        raise ValueError("IDs must be unique in test.csv and sample_submission.csv.")
    if len(test) != len(sample) or set(test["id"]) != set(sample["id"]):
        raise ValueError("IDs in test.csv and sample_submission.csv must match.")
    expected = set(train.columns) - {"is_fraud"}
    if set(test.columns) != expected:
        raise ValueError("test.csv columns do not match training columns.")
    test = test.set_index("id").loc[sample["id"]].reset_index()
    return train, test, sample


def _predict_fold(
    version: str, model_name: str, fold: int, train: pd.DataFrame, test: pd.DataFrame
) -> np.ndarray:
    model_dir = ROOT / "models" / f"ml_{version}"
    pipeline_path = model_dir / f"{model_name}_features_fold_{fold}.joblib"
    if not pipeline_path.is_file():
        raise FileNotFoundError(f"Missing fitted feature pipeline: {pipeline_path}")
    feature_pipeline = joblib.load(pipeline_path)
    X_test = feature_pipeline.transform(test)

    if model_name == "logistic_regression":
        model_path = model_dir / f"{model_name}_fold_{fold}.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model: {model_path}")
        model = joblib.load(model_path)
        bool_columns = X_test.select_dtypes(include=["bool"]).columns
        if len(bool_columns):
            X_test[bool_columns] = X_test[bool_columns].astype("int8")
        return model.predict_proba(X_test)[:, 1]

    if model_name == "catboost":
        from catboost import CatBoostClassifier

        model_path = model_dir / f"{model_name}_fold_{fold}.cbm"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model: {model_path}")
        categorical_columns = X_test.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()
        for column in categorical_columns:
            X_test[column] = X_test[column].astype(object).fillna("__UNKNOWN__")
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        return model.predict_proba(X_test)[:, 1]

    if model_name == "lightgbm":
        from lightgbm import Booster

        model_path = model_dir / f"{model_name}_fold_{fold}.txt"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model: {model_path}")
        model = Booster(model_file=str(model_path))
        categorical_columns = X_test.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()
        categories = model.pandas_categorical or []
        if len(categories) != len(categorical_columns):
            raise ValueError(f"LightGBM fold {fold} categorical schema mismatch.")
        for column, category_values in zip(categorical_columns, categories):
            X_test[column] = pd.Categorical(X_test[column], categories=category_values)
        return np.asarray(model.predict(X_test), dtype=np.float64)

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        model_path = model_dir / f"{model_name}_fold_{fold}.json"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model: {model_path}")
        model = XGBClassifier()
        model.load_model(str(model_path))
        return model.predict_proba(X_test)[:, 1]

    raise ValueError(f"Unknown model: {model_name}")


def _save_predictions(
    version: str, model_name: str, probabilities: np.ndarray, sample: pd.DataFrame
) -> None:
    if len(probabilities) != len(sample) or not np.isfinite(probabilities).all():
        raise ValueError(f"{model_name} produced invalid predictions.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{model_name} predictions must be in [0, 1].")

    version_name = f"ml_{version}"
    prediction_dir = ROOT / "predictions" / "test" / version_name
    submission_dir = ROOT / "submissions" / version_name
    prediction_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"id": sample["id"], "prediction": probabilities}).to_csv(
        prediction_dir / f"{model_name}_test.csv", index=False
    )
    submission = sample.copy()
    submission["is_fraud"] = probabilities
    submission.to_csv(submission_dir / f"submission_{model_name}.csv", index=False)
    print(
        f"{version}/{model_name}: saved predictions and submission "
        f"({len(probabilities):,} rows)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("all", "v1", "v2"), required=True)
    parser.add_argument(
        "--model",
        choices=("all", "logistic_regression", "catboost", "lightgbm", "xgboost"),
        default="all",
    )
    args = parser.parse_args()
    train, test, sample = _load_data()
    versions = ("v1", "v2") if args.version == "all" else (args.version,)
    models = (
        ("logistic_regression", "catboost", "lightgbm", "xgboost")
        if args.model == "all"
        else (args.model,)
    )

    for version in versions:
        for model_name in models:
            model_dir = ROOT / "models" / f"ml_{version}"
            required = [
                model_dir / f"{model_name}_fold_{fold}.{ {'logistic_regression': 'joblib', 'catboost': 'cbm', 'lightgbm': 'txt', 'xgboost': 'json'}[model_name] }"
                for fold in range(N_FOLDS)
            ]
            if not all(path.is_file() for path in required):
                print(f"Skipping {version}/{model_name}: five fold models are not all present.")
                continue
            fold_predictions = [
                _predict_fold(version, model_name, fold, train, test)
                for fold in range(N_FOLDS)
            ]
            _save_predictions(version, model_name, np.mean(fold_predictions, axis=0), sample)


if __name__ == "__main__":
    main()
