"""Create Kaggle submissions from the saved 5-fold ML baseline models.

Run from the project root:
    .\\.venv\\Scripts\\python.exe ml\\trainers\\predict_ml_baselines.py

Each submission averages the five fold models and follows the ID order in
data/raw/sample_submission.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
TEST_PATH = ROOT / "data" / "raw" / "test.csv"
SAMPLE_PATH = ROOT / "data" / "raw" / "sample_submission.csv"
N_FOLDS = 5


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    if list(sample.columns) != ["id", "is_fraud"]:
        raise ValueError("sample_submission.csv must contain id,is_fraud columns.")
    if not test["id"].is_unique or not sample["id"].is_unique:
        raise ValueError("IDs must be unique in test.csv and sample_submission.csv.")
    if len(test) != len(sample) or set(test["id"]) != set(sample["id"]):
        raise ValueError("IDs in test.csv and sample_submission.csv must match.")

    feature_columns = [c for c in train.columns if c not in {"id", "is_fraud"}]
    if set(test.columns) != set(feature_columns) | {"id"}:
        raise ValueError("test.csv columns do not match the training features.")
    test_features = test.set_index("id").loc[sample["id"], feature_columns]
    return train, test_features, sample, feature_columns


def _validate_predictions(predictions: np.ndarray, model_name: str) -> None:
    if len(predictions) == 0 or not np.isfinite(predictions).all():
        raise ValueError(f"{model_name} produced empty or non-finite predictions.")
    if ((predictions < 0) | (predictions > 1)).any():
        raise ValueError(f"{model_name} predictions must be probabilities in [0, 1].")


def _predict_logistic(
    train: pd.DataFrame, test_features: pd.DataFrame, feature_columns: list[str]
) -> np.ndarray:
    import joblib

    predictions = np.zeros(len(test_features), dtype=np.float64)
    for fold in range(N_FOLDS):
        path = MODEL_DIR / f"logistic_regression_fold_{fold}.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Logistic Regression model: {path}")
        model = joblib.load(path)
        fold_features = test_features.copy()
        bool_columns = train[feature_columns].select_dtypes(include=["bool"]).columns
        if len(bool_columns):
            fold_features[bool_columns] = fold_features[bool_columns].astype("int8")
        predictions += model.predict_proba(fold_features)[:, 1] / N_FOLDS
    return predictions


def _predict_catboost(
    train: pd.DataFrame, test_features: pd.DataFrame
) -> np.ndarray:
    from catboost import CatBoostClassifier

    categorical_columns = train.drop(columns=["id", "is_fraud"]).select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    if test_features[categorical_columns].isna().any().any():
        raise ValueError("CatBoost categorical test features contain missing values.")

    predictions = np.zeros(len(test_features), dtype=np.float64)
    for fold in range(N_FOLDS):
        path = MODEL_DIR / f"catboost_fold_{fold}.cbm"
        if not path.is_file():
            raise FileNotFoundError(f"Missing CatBoost model: {path}")
        model = CatBoostClassifier()
        model.load_model(str(path))
        predictions += model.predict_proba(test_features)[:, 1] / N_FOLDS
    return predictions


def _predict_lightgbm(
    train: pd.DataFrame, test_features: pd.DataFrame
) -> np.ndarray:
    from lightgbm import Booster

    categorical_columns = train.drop(columns=["id", "is_fraud"]).select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    predictions = np.zeros(len(test_features), dtype=np.float64)
    for fold in range(N_FOLDS):
        path = MODEL_DIR / f"lightgbm_fold_{fold}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing LightGBM model: {path}")
        model = Booster(model_file=str(path))
        fold_features = test_features.copy()
        categories = model.pandas_categorical or []
        if len(categories) != len(categorical_columns):
            raise ValueError(f"LightGBM fold {fold} categorical schema mismatch.")
        for column, category_values in zip(categorical_columns, categories):
            fold_features[column] = pd.Categorical(
                fold_features[column], categories=category_values
            )
        predictions += np.asarray(model.predict(fold_features), dtype=np.float64) / N_FOLDS
    return predictions


def _predict_xgboost(
    train: pd.DataFrame, test_features: pd.DataFrame
) -> np.ndarray:
    from xgboost import XGBClassifier

    categorical_columns = train.drop(columns=["id", "is_fraud"]).select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    folds = pd.read_csv(ROOT / "validation" / "folds.csv").set_index("id")["fold"]
    row_folds = train["id"].map(folds)
    if row_folds.isna().any():
        raise ValueError("Could not match every training ID to a validation fold.")

    predictions = np.zeros(len(test_features), dtype=np.float64)
    for fold in range(N_FOLDS):
        path = MODEL_DIR / f"xgboost_fold_{fold}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing XGBoost model: {path}")
        fold_features = test_features.copy()
        for column in categorical_columns:
            categories = train.loc[row_folds != fold, column].astype("category").cat.categories
            fold_features[column] = pd.Categorical(
                fold_features[column], categories=categories
            )
        model = XGBClassifier()
        model.load_model(str(path))
        predictions += model.predict_proba(fold_features)[:, 1] / N_FOLDS
    return predictions


def _save_submission(
    model_name: str, predictions: np.ndarray, sample: pd.DataFrame
) -> None:
    _validate_predictions(predictions, model_name)

    prediction_dir = ROOT / "predictions" / "test"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    test_predictions = pd.DataFrame(
        {"id": sample["id"], "prediction": predictions}
    )
    prediction_path = prediction_dir / f"{model_name}_test.csv"
    test_predictions.to_csv(prediction_path, index=False)
    print(f"Saved {prediction_path}")

    output_dir = ROOT / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)
    submission = sample.copy()
    submission["is_fraud"] = predictions
    output_path = output_dir / f"submission_{model_name}.csv"
    submission.to_csv(output_path, index=False)
    print(f"Saved {output_path}")
    print(
        f"Rows: {len(submission):,}; probability range: "
        f"[{predictions.min():.6f}, {predictions.max():.6f}]"
    )


def main() -> None:
    train, test_features, sample, feature_columns = _load_data()
    _save_submission(
        "logistic_regression",
        _predict_logistic(train, test_features, feature_columns),
        sample,
    )

    catboost_models = [MODEL_DIR / f"catboost_fold_{fold}.cbm" for fold in range(N_FOLDS)]
    if all(path.is_file() for path in catboost_models):
        _save_submission("catboost", _predict_catboost(train, test_features), sample)
    else:
        print("Skipping CatBoost: its five fold models have not all been trained.")

    lightgbm_models = [MODEL_DIR / f"lightgbm_fold_{fold}.txt" for fold in range(N_FOLDS)]
    if all(path.is_file() for path in lightgbm_models):
        _save_submission("lightgbm", _predict_lightgbm(train, test_features), sample)
    else:
        print("Skipping LightGBM: its five fold models have not all been trained.")

    xgboost_models = [MODEL_DIR / f"xgboost_fold_{fold}.json" for fold in range(N_FOLDS)]
    if all(path.is_file() for path in xgboost_models):
        _save_submission("xgboost", _predict_xgboost(train, test_features), sample)
    else:
        print("Skipping XGBoost: its five fold models have not all been trained.")


if __name__ == "__main__":
    main()
