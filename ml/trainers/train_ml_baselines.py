"""Run all ML baselines with the shared five-fold assignment.

Run from the project root:
    .\\.venv\\Scripts\\python.exe -m ml.trainers.train_ml_baselines

All columns in data/raw/train.csv are used as features except ``id`` and
``is_fraud``. Both models use the shared validation/folds.csv assignment.
"""

import sys
from pathlib import Path

# Make imports work both when this file is launched directly from an IDE and
# when it is run as ``python -m ml.trainers.train_ml_baselines``.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.trainers.train_catboost import train_catboost
from ml.trainers.train_lightgbm import train_lightgbm
from ml.trainers.train_logistic_regression import train_logistic_regression
from ml.trainers.train_xgboost import train_xgboost


def main() -> None:
    print("=== Logistic Regression baseline ===", flush=True)
    train_logistic_regression()

    print("\n=== CatBoost baseline ===", flush=True)
    train_catboost()

    print("\n=== LightGBM baseline ===", flush=True)
    train_lightgbm()

    print("\n=== XGBoost baseline ===", flush=True)
    train_xgboost()


if __name__ == "__main__":
    main()
