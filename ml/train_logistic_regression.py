from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.logistic_regression_model import build_model

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
FOLDS_PATH = ROOT / "validation" / "folds.csv"
N_FOLDS = 5
RANDOM_STATE = 42


def train_logistic_regression() -> None:
    if not FOLDS_PATH.is_file():
        raise FileNotFoundError(
            "Missing validation/folds.csv. Run "
            "`python validation\\make_folds.py` first."
        )

    train = pd.read_csv(TRAIN_PATH)
    folds = pd.read_csv(FOLDS_PATH)

    required_train_columns = {"id", "is_fraud"}
    required_fold_columns = {"id", "fold"}
    if not required_train_columns.issubset(train.columns):
        raise ValueError("train.csv must contain id and is_fraud columns.")
    if not required_fold_columns.issubset(folds.columns):
        raise ValueError("folds.csv must contain id and fold columns.")
    if train["id"].duplicated().any() or folds["id"].duplicated().any():
        raise ValueError("IDs must be unique in train.csv and folds.csv.")
    if set(train["id"]) != set(folds["id"]):
        raise ValueError("IDs in folds.csv must match IDs in train.csv.")

    data = train.merge(
        folds[["id", "fold"]],
        on="id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if data["fold"].isna().any():
        raise ValueError("Every training row must have a fold assignment.")
    if set(data["fold"].unique()) != set(range(N_FOLDS)):
        raise ValueError(f"folds.csv must contain folds 0 through {N_FOLDS - 1}.")

    feature_columns = [
        column for column in train.columns if column not in {"id", "is_fraud"}
    ]
    features = data[feature_columns]
    boolean_columns = features.select_dtypes(include=["bool"]).columns.tolist()
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = features.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    classified_columns = (
        set(boolean_columns) | set(numeric_columns) | set(categorical_columns)
    )
    unclassified_columns = set(feature_columns) - classified_columns
    if unclassified_columns:
        raise ValueError(
            f"Unsupported feature types: {sorted(unclassified_columns)}"
        )

    labels = data["is_fraud"].astype(int)
    fold_results: list[dict[str, float | int]] = []
    oof_parts: list[pd.DataFrame] = []

    for fold in range(N_FOLDS):
        validation_mask = data["fold"] == fold
        training_mask = ~validation_mask
        model = build_model(
            numeric_columns,
            categorical_columns,
            boolean_columns,
            random_state=RANDOM_STATE,
        )
        model.fit(features.loc[training_mask], labels.loc[training_mask])

        validation_features = features.loc[validation_mask]
        validation_labels = labels.loc[validation_mask]
        predictions = model.predict_proba(validation_features)[:, 1]
        fold_auc = float(roc_auc_score(validation_labels, predictions))
        fold_results.append({"fold": fold, "roc_auc": fold_auc})
        oof_parts.append(
            pd.DataFrame(
                {
                    "id": data.loc[validation_mask, "id"].to_numpy(),
                    "fold": fold,
                    "y_true": validation_labels.to_numpy(),
                    "oof_prediction": predictions,
                }
            )
        )

        model_path = ROOT / "models" / f"logistic_regression_fold_{fold}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Fold {fold}: ROC AUC={fold_auc:.6f}")

    results = pd.DataFrame(fold_results)
    oof = pd.concat(oof_parts, ignore_index=True)
    if len(oof) != len(train) or oof["id"].duplicated().any():
        raise RuntimeError("OOF output must contain exactly one row per training ID.")

    evaluation_dir = ROOT / "evaluation"
    predictions_dir = ROOT / "predictions" / "oof"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        evaluation_dir / "logistic_regression_cv_results.csv", index=False
    )
    oof.to_csv(predictions_dir / "logistic_regression_oof.csv", index=False)

    print(f"Mean ROC AUC: {results['roc_auc'].mean():.6f}")
    print(f"Std ROC AUC:  {results['roc_auc'].std(ddof=1):.6f}")
    print(f"OOF ROC AUC:  {roc_auc_score(oof['y_true'], oof['oof_prediction']):.6f}")
    print("Saved fold models, CV results, and OOF predictions.")


if __name__ == "__main__":
    train_logistic_regression()
