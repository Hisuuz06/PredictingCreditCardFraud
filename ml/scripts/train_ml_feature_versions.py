"""Train ML models using FeaturePipeline v1 and/or v2.

Run from the project root:
    python -m ml.scripts.train_ml_feature_versions
    python -m ml.scripts.train_ml_feature_versions --version v1
    python -m ml.scripts.train_ml_feature_versions --version v2

Each version uses the shared five-fold assignment. Models, CV metrics, and OOF
predictions are kept in version-specific directories (ml_v1 / ml_v2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.trainers.train_catboost import train_catboost
from ml.trainers.train_lightgbm import train_lightgbm
from ml.trainers.train_logistic_regression import train_logistic_regression
from ml.trainers.train_xgboost import train_xgboost


TRAINERS = (
    ("Logistic Regression", train_logistic_regression),
    ("CatBoost", train_catboost),
    ("LightGBM", train_lightgbm),
    ("XGBoost", train_xgboost),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", choices=("all", "v1", "v2"), default="all",
        help="FeaturePipeline version to train (default: both v1 and v2).",
    )
    args = parser.parse_args()

    train_columns = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()
    if not {"id", "is_fraud"}.issubset(train_columns):
        raise ValueError("train.csv must contain id and is_fraud columns.")
    feature_columns = [c for c in train_columns if c not in {"id", "is_fraud"}]
    versions = ("v1", "v2") if args.version == "all" else (args.version,)

    for version in versions:
        print(f"\n========== FeaturePipeline {version} ==========", flush=True)
        for display_name, trainer in TRAINERS:
            print(f"\n=== {display_name} ({version}) ===", flush=True)
            trainer(feature_columns, feature_version=version)


if __name__ == "__main__":
    main()
