"""Run all ML baselines with the shared five-fold assignment.

Run from the project root:
    .\\.venv\\Scripts\\python.exe -m ml.scripts.train_ml_baselines

All columns in data/raw/train.csv are used as features except ``id`` and
``is_fraud``. All four models use the shared validation/folds.csv assignment.
Models are written to models/ml_baseline/; CV results and OOF predictions
are written under evaluation/ml_baseline/ and predictions/ml_baseline/.
"""

import sys
from pathlib import Path

import pandas as pd

# Make imports work both when this file is launched directly from an IDE and
# when it is run as ``python -m ml.scripts.train_ml_baselines``.
ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.trainers.train_catboost import train_catboost
from ml.trainers.train_lightgbm import train_lightgbm
from ml.trainers.train_logistic_regression import train_logistic_regression
from ml.trainers.train_xgboost import train_xgboost


def main() -> None:
    train_columns = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()
    if not {"id", "is_fraud"}.issubset(train_columns):
        raise ValueError("train.csv must contain id and is_fraud columns.")
    # This script owns the model input definition. Edit this list to select a
    # different set of features; currently it uses every column except ID/target.
    feature_columns = [
        column for column in train_columns if column not in {"id", "is_fraud"}
    ]

    print("=== Logistic Regression baseline ===", flush=True)
    train_logistic_regression(feature_columns)

    print("\n=== CatBoost baseline ===", flush=True)
    train_catboost(feature_columns)

    print("\n=== LightGBM baseline ===", flush=True)
    train_lightgbm(feature_columns)

    print("\n=== XGBoost baseline ===", flush=True)
    train_xgboost(feature_columns)


if __name__ == "__main__":
    main()
