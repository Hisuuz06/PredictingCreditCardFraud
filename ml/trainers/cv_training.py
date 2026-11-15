from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
FOLDS_PATH = ROOT / "validation" / "folds.csv"
N_FOLDS = 5

ModelFactory = Callable[[pd.DataFrame, pd.Series, list[str]], Any]
FitPredict = Callable[
    [Any, pd.DataFrame, pd.Series, pd.DataFrame, list[str]],
    np.ndarray,
]
SaveModel = Callable[[Any, Path], None]


def train_cross_validation(
    model_name: str,
    model_extension: str,
    model_factory: ModelFactory,
    fit_predict: FitPredict,
    save_model: SaveModel,
) -> None:
    if not FOLDS_PATH.is_file():
        raise FileNotFoundError(
            "Missing validation/folds.csv. Run "
            "`python validation\\make_folds.py` first."
        )

    train = pd.read_csv(TRAIN_PATH)
    folds = pd.read_csv(FOLDS_PATH)
    if not {"id", "is_fraud"}.issubset(train.columns):
        raise ValueError("train.csv must contain id and is_fraud columns.")
    if not {"id", "fold"}.issubset(folds.columns):
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
    categorical_columns = features.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    labels = data["is_fraud"].astype(int)

    fold_results: list[dict[str, float | int]] = []
    oof_parts: list[pd.DataFrame] = []
    for fold in range(N_FOLDS):
        validation_mask = data["fold"] == fold
        training_mask = ~validation_mask
        y_train = labels.loc[training_mask]
        model = model_factory(
            features.loc[training_mask],
            y_train,
            categorical_columns,
        )
        predictions = np.asarray(
            fit_predict(
                model,
                features.loc[training_mask],
                y_train,
                features.loc[validation_mask],
                categorical_columns,
            ),
            dtype=np.float64,
        ).reshape(-1)
        y_validation = labels.loc[validation_mask]
        if len(predictions) != len(y_validation):
            raise RuntimeError(f"Fold {fold} returned the wrong number of predictions.")
        if not np.isfinite(predictions).all() or (
            (predictions < 0) | (predictions > 1)
        ).any():
            raise RuntimeError(f"Fold {fold} predictions must be finite probabilities.")

        fold_auc = float(roc_auc_score(y_validation, predictions))
        fold_results.append({"fold": fold, "roc_auc": fold_auc})
        oof_parts.append(
            pd.DataFrame(
                {
                    "id": data.loc[validation_mask, "id"].to_numpy(),
                    "fold": fold,
                    "y_true": y_validation.to_numpy(),
                    "oof_prediction": predictions,
                }
            )
        )

        model_path = ROOT / "models" / f"{model_name}_fold_{fold}.{model_extension}"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save_model(model, model_path)
        print(f"Fold {fold}: ROC AUC={fold_auc:.6f}")

    results = pd.DataFrame(fold_results)
    oof = pd.concat(oof_parts, ignore_index=True)
    if len(oof) != len(train) or oof["id"].duplicated().any():
        raise RuntimeError("OOF output must contain exactly one row per training ID.")

    evaluation_dir = ROOT / "evaluation"
    predictions_dir = ROOT / "predictions" / "oof"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(evaluation_dir / f"{model_name}_cv_results.csv", index=False)
    oof.to_csv(predictions_dir / f"{model_name}_oof.csv", index=False)

    print(f"Mean ROC AUC: {results['roc_auc'].mean():.6f}")
    print(f"Std ROC AUC:  {results['roc_auc'].std(ddof=1):.6f}")
    print(f"OOF ROC AUC:  {roc_auc_score(oof['y_true'], oof['oof_prediction']):.6f}")
    print(f"Saved {model_name} fold models, CV results, and OOF predictions.")
