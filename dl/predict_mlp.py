r"""Dự đoán test bằng 5 MLP folds và tạo submission cho Kaggle.

Chạy từ thư mục gốc dự án:
    .\.venv\Scripts\python.exe dl\predict_mlp.py

Cần có: data/raw/test.csv, data/raw/sample_submission.csv,
         models/mlp_fold{0..4}.pt và models/mlp_fold{0..4}_preprocessor.joblib.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5
BATCH_SIZE = 4096


def build_model(input_dim: int) -> nn.Module:
    """Phải giống hệt kiến trúc đã dùng trong train_mlp_baseline.py."""
    return nn.Sequential(
        nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
    )


def predict_one_fold(fold: int, test: pd.DataFrame) -> np.ndarray:
    """Dùng đúng preprocessor + trọng số được lưu từ fold tương ứng."""
    model_path = MODEL_DIR / f"mlp_fold{fold}.pt"
    preprocessor_path = MODEL_DIR / f"mlp_fold{fold}_preprocessor.joblib"
    if not model_path.is_file() or not preprocessor_path.is_file():
        raise FileNotFoundError(f"Thiếu artifact của fold {fold} trong thư mục models/")

    preprocessor = joblib.load(preprocessor_path)
    features = list(preprocessor.feature_names_in_)
    if set(test.columns) != set(features):
        missing = sorted(set(features) - set(test.columns))
        extra = sorted(set(test.columns) - set(features))
        raise ValueError(f"Các cột test không khớp fold {fold}: thiếu={missing}, dư={extra}")

    # Không fit lại trên test! Chỉ transform bằng bộ đã học ở training fold.
    x_test = np.asarray(preprocessor.transform(test[features]), dtype=np.float32).copy()
    if not np.isfinite(x_test).all():
        raise ValueError(f"Feature test ở fold {fold} chứa NaN/Inf")

    # Chỉ load các tensor trọng số, không load đối tượng Python tùy ý.
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    input_dim = int(checkpoint["input_dim"])
    if x_test.shape[1] != input_dim:
        raise ValueError(f"Fold {fold}: số feature sau mã hóa không khớp checkpoint")

    model = build_model(input_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(DEVICE)
    model.eval()  # Tắt Dropout khi dự đoán

    loader = DataLoader(TensorDataset(torch.from_numpy(x_test)), batch_size=BATCH_SIZE)
    results = []
    with torch.inference_mode():
        for (batch_x,) in loader:
            logits = model(batch_x.to(DEVICE))
            probabilities = torch.sigmoid(logits).cpu().numpy().ravel()
            results.append(probabilities)
    return np.concatenate(results)


def main() -> None:
    test = pd.read_csv(ROOT / "data" / "raw" / "test.csv")
    sample = pd.read_csv(ROOT / "data" / "raw" / "sample_submission.csv")

    # Giữ đúng ID và thứ tự mà Kaggle yêu cầu trong sample_submission.csv.
    if list(sample.columns) != ["id", "is_fraud"]:
        raise ValueError("sample_submission.csv phải có đúng hai cột: id,is_fraud")
    if not test["id"].is_unique or not sample["id"].is_unique:
        raise ValueError("ID bị trùng ở test hoặc sample_submission")
    if len(test) != len(sample) or set(test["id"]) != set(sample["id"]):
        raise ValueError("ID của test.csv và sample_submission.csv không khớp")

    test_features = test.set_index("id").loc[sample["id"]]
    predictions = np.zeros(len(sample), dtype=np.float64)

    for fold in range(N_FOLDS):
        fold_preds = predict_one_fold(fold, test_features)
        predictions += fold_preds / N_FOLDS
        print(f"Fold {fold}: hoàn tất {len(fold_preds):,} dự đoán")

    if not np.isfinite(predictions).all() or not ((0 <= predictions) & (predictions <= 1)).all():
        raise ValueError("Dự đoán chứa NaN/Inf hoặc ngoài khoảng [0,1]")

    pred_dir = ROOT / "predictions" / "test"
    submit_dir = ROOT / "submissions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    submit_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"id": sample["id"], "prediction": predictions}).to_csv(
        pred_dir / "mlp_test.csv", index=False
    )
    submission = sample.copy()
    submission["is_fraud"] = predictions
    output = submit_dir / "submission_baseline.csv"
    submission.to_csv(output, index=False)

    print(f"\nĐã tạo: {output}")
    print(f"Số dòng: {len(submission):,}; xác suất min={predictions.min():.6f}, "
          f"max={predictions.max():.6f}; device={DEVICE}")
    print(submission.head().to_string(index=False))
    print("Hoàn thành")


if __name__ == "__main__":
    main()
