"""Predict test probabilities from saved 5-fold MLP checkpoints.

Run from the project root:
    .\\.venv\\Scripts\\python.exe -m dl.scripts.predict_mlp --variant baseline
    .\\.venv\\Scripts\\python.exe -m dl.scripts.predict_mlp --variant v3_linear
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dl.models.mlp_model import build_mlp_model

TEST_PATH = ROOT / "data" / "raw" / "test.csv"
SAMPLE_PATH = ROOT / "data" / "raw" / "sample_submission.csv"
N_FOLDS = 5
BATCH_SIZE = 4096
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_dense_float32(matrix: object) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32).copy()


def _predict_one_fold(
    fold: int,
    test_features: pd.DataFrame,
    model_dir: Path,
    *,
    variant: str,
) -> np.ndarray:
    model_path = model_dir / f"mlp_fold{fold}.pt"
    preprocessor_path = model_dir / f"mlp_fold{fold}_preprocessor.joblib"
    if not model_path.is_file() or not preprocessor_path.is_file():
        raise FileNotFoundError(f"Missing fold {fold} artifacts in {model_dir}")

    features_df = test_features
    if variant == "v3_linear":
        fp_path = model_dir / f"mlp_features_fold{fold}.joblib"
        if not fp_path.is_file():
            raise FileNotFoundError(f"Missing feature pipeline for fold {fold}: {fp_path}")
        feature_pipeline = joblib.load(fp_path)
        features_df = feature_pipeline.transform(test_features)

    preprocessor = joblib.load(preprocessor_path)
    expected_cols = list(preprocessor.feature_names_in_)
    if set(features_df.columns) != set(expected_cols):
        missing = sorted(set(expected_cols) - set(features_df.columns))
        extra = sorted(set(features_df.columns) - set(expected_cols))
        raise ValueError(f"Fold {fold} test columns mismatch: missing={missing}, extra={extra}")

    x_test = _to_dense_float32(preprocessor.transform(features_df[expected_cols]))
    if not np.isfinite(x_test).all():
        raise ValueError(f"Fold {fold} test features contain NaN/Inf.")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    input_dim = int(checkpoint["input_dim"])
    hidden_dims = tuple(checkpoint.get("hidden_dims", (64, 32)))
    dropout = float(checkpoint.get("dropout", 0.2))
    if x_test.shape[1] != input_dim:
        raise ValueError(f"Fold {fold}: feature count {x_test.shape[1]} != checkpoint {input_dim}")

    model = build_mlp_model(input_dim, hidden_dims=hidden_dims, dropout=dropout)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(DEVICE)
    model.eval()

    loader = DataLoader(TensorDataset(torch.from_numpy(x_test)), batch_size=BATCH_SIZE)
    results = []
    with torch.inference_mode():
        for (batch_x,) in loader:
            probs = torch.sigmoid(model(batch_x.to(DEVICE))).cpu().numpy().ravel()
            results.append(probs)
    return np.concatenate(results)


def predict_mlp(variant: str = "baseline") -> None:
    experiment_name = f"dl_{variant}"
    model_dir = ROOT / "models" / experiment_name

    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    if list(sample.columns) != ["id", "is_fraud"]:
        raise ValueError("sample_submission.csv must contain id,is_fraud columns.")
    if not test["id"].is_unique or not sample["id"].is_unique:
        raise ValueError("IDs must be unique in test.csv and sample_submission.csv.")
    if len(test) != len(sample) or set(test["id"]) != set(sample["id"]):
        raise ValueError("IDs in test.csv and sample_submission.csv must match.")

    test_features = test.set_index("id").loc[sample["id"]]
    predictions = np.zeros(len(sample), dtype=np.float64)

    for fold in range(N_FOLDS):
        fold_preds = _predict_one_fold(fold, test_features, model_dir, variant=variant)
        predictions += fold_preds / N_FOLDS
        print(f"[{experiment_name}] Fold {fold}: predicted {len(fold_preds):,} rows")

    if not np.isfinite(predictions).all() or not ((0 <= predictions) & (predictions <= 1)).all():
        raise ValueError("Predictions contain NaN/Inf or values outside [0, 1].")

    pred_dir = ROOT / "predictions" / "test" / experiment_name
    submit_dir = ROOT / "submissions" / experiment_name
    pred_dir.mkdir(parents=True, exist_ok=True)
    submit_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"id": sample["id"], "prediction": predictions}).to_csv(
        pred_dir / "mlp_test.csv", index=False
    )
    submission = sample.copy()
    submission["is_fraud"] = predictions
    sub_path = submit_dir / "submission_mlp.csv"
    submission.to_csv(sub_path, index=False)

    print(f"\nSaved: {pred_dir / 'mlp_test.csv'}")
    print(f"Saved: {sub_path}")
    print(f"Rows={len(submission):,} | min={predictions.min():.6f} | max={predictions.max():.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict test probabilities with 5-fold MLP.")
    parser.add_argument(
        "--variant",
        default="baseline",
        choices=["baseline", "v3_linear"],
        help="Model variant to predict with.",
    )
    args = parser.parse_args()
    predict_mlp(variant=args.variant)


if __name__ == "__main__":
    main()
