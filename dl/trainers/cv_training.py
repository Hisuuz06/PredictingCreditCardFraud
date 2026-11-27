from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dl.models.mlp_model import build_mlp_model, build_mlp_preprocessor
from features.feature_pipeline import FeaturePipeline

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
FOLDS_PATH = ROOT / "validation" / "folds.csv"
N_FOLDS = 5


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_dense_float32(matrix: object) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32).copy()


def _load_training_data() -> pd.DataFrame:
    if not FOLDS_PATH.is_file():
        raise FileNotFoundError(
            "Missing validation/folds.csv. Run `python validation\\make_folds.py` first."
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
    return data


def train_mlp_cross_validation(
    *,
    variant: str = "baseline",
    epochs: int = 5,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    hidden_dims: tuple[int, ...] = (64, 32),
    dropout: float = 0.2,
    seed: int = 42,
) -> None:
    """Train 5-fold MLP CV and save models, OOF predictions, and CV metrics."""
    experiment_name = f"dl_{variant}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _load_training_data()
    feature_columns = [c for c in data.columns if c not in ("id", "is_fraud", "fold")]

    model_dir = ROOT / "models" / experiment_name
    eval_dir = ROOT / "evaluation" / experiment_name
    oof_dir = ROOT / "predictions" / "oof" / experiment_name
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    oof_dir.mkdir(parents=True, exist_ok=True)

    fold_results: list[dict[str, float | int]] = []
    oof_parts: list[pd.DataFrame] = []

    for fold in range(N_FOLDS):
        _set_seed(seed)
        val_mask = data["fold"] == fold
        train_df = data.loc[~val_mask]
        val_df = data.loc[val_mask]

        if variant == "v3_linear":
            fp_cols = ["id", *feature_columns, "is_fraud"]
            fp = FeaturePipeline(version="v1").fit(train_df[fp_cols])
            x_train_raw = fp.transform(train_df[fp_cols])
            x_val_raw = fp.transform(val_df[fp_cols])
            joblib.dump(fp, model_dir / f"mlp_features_fold{fold}.joblib")
        else:
            x_train_raw = train_df[feature_columns]
            x_val_raw = val_df[feature_columns]

        num_cols = x_train_raw.select_dtypes(include="number").columns.tolist()
        cat_cols = x_train_raw.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()
        bool_cols = x_train_raw.select_dtypes(include="bool").columns.tolist()

        preprocessor = build_mlp_preprocessor(
            num_cols, cat_cols, bool_cols, variant=variant
        )
        x_train = _to_dense_float32(preprocessor.fit_transform(x_train_raw))
        x_val = _to_dense_float32(preprocessor.transform(x_val_raw))
        if not np.isfinite(x_train).all() or not np.isfinite(x_val).all():
            raise ValueError(f"Fold {fold} preprocessed features contain NaN/Inf.")

        y_train = train_df["is_fraud"].to_numpy(dtype=np.float32, copy=True)
        y_val = val_df["is_fraud"].to_numpy(dtype=np.int32, copy=True)

        model = build_mlp_model(
            x_train.shape[1], hidden_dims=hidden_dims, dropout=dropout
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train).unsqueeze(1)),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val)), batch_size=4096)

        best_auc, best_epoch = -1.0, 0
        best_weights = None
        best_predictions = None
        print(
            f"\n[{experiment_name}] FOLD {fold} | train={len(train_df):,} | "
            f"val={len(val_df):,} | features={x_train.shape[1]} | device={device}"
        )

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_x)

            model.eval()
            preds = []
            with torch.inference_mode():
                for (batch_x,) in val_loader:
                    probs = torch.sigmoid(model(batch_x.to(device)))
                    preds.append(probs.cpu().numpy().ravel())
            val_preds = np.concatenate(preds)
            auc = float(roc_auc_score(y_val, val_preds))
            print(
                f"  Epoch {epoch}/{epochs} | loss={total_loss / len(train_df):.5f} | AUC={auc:.6f}"
            )

            if auc > best_auc:
                best_auc, best_epoch = auc, epoch
                best_weights = copy.deepcopy(model.state_dict())
                best_predictions = val_preds.copy()

        torch.save(
            {
                "state_dict": best_weights,
                "input_dim": x_train.shape[1],
                "hidden_dims": list(hidden_dims),
                "dropout": dropout,
            },
            model_dir / f"mlp_fold{fold}.pt",
        )
        joblib.dump(preprocessor, model_dir / f"mlp_fold{fold}_preprocessor.joblib")

        fold_oof = pd.DataFrame(
            {
                "id": val_df["id"].to_numpy(),
                "fold": fold,
                "y_true": y_val,
                "oof_prediction": best_predictions,
            }
        )
        fold_oof.to_csv(oof_dir / f"mlp_fold{fold}_oof.csv", index=False)
        fold_results.append(
            {"fold": fold, "best_epoch": best_epoch, "roc_auc": best_auc}
        )
        oof_parts.append(fold_oof)

    scores_df = pd.DataFrame(fold_results)
    scores_df.to_csv(eval_dir / "mlp_cv_results.csv", index=False)

    oof = pd.concat(oof_parts, ignore_index=True).sort_values("id")
    if len(oof) != len(data) or oof["id"].duplicated().any():
        raise RuntimeError("OOF output must contain exactly one row per training ID.")
    oof.to_csv(oof_dir / "mlp_oof.csv", index=False)

    mean_auc = float(scores_df["roc_auc"].mean())
    std_auc = float(scores_df["roc_auc"].std(ddof=1))
    oof_auc = float(roc_auc_score(oof["y_true"], oof["oof_prediction"]))

    meta = {
        "model": "MLP",
        "variant": variant,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_dims": list(hidden_dims),
        "dropout": dropout,
        "cv_auc_mean": round(mean_auc, 6),
        "cv_auc_std": round(std_auc, 6),
        "oof_auc": round(oof_auc, 6),
    }
    with open(eval_dir / "mlp_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n========== {experiment_name} 5-FOLD RESULTS ==========")
    print(scores_df.to_string(index=False))
    print(f"Mean ROC AUC: {mean_auc:.6f}")
    print(f"Std ROC AUC : {std_auc:.6f}")
    print(f"OOF ROC AUC : {oof_auc:.6f}")
