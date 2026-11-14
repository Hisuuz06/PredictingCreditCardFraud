r"""MLP baseline: chạy 5-fold CV và lưu OOF.

Chạy từ thư mục gốc dự án:
    .\.venv\Scripts\python.exe dl\train_mlp_baseline.py

Yêu cầu: data/raw/train.csv và validation/folds.csv đã có sẵn.
"""
from pathlib import Path
import copy
import random

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ===== 1. Cấu hình =====
ROOT = Path(__file__).resolve().parents[1]
SEED = 42
EPOCHS = 5
BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_fold(data: pd.DataFrame, fold: int):
    """Huấn luyện bằng 4 folds, đánh giá trên fold được chọn."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ===== 2. Chia train / validation =====
    train_df = data[data["fold"] != fold]
    val_df = data[data["fold"] == fold]
    feature_cols = [c for c in data.columns if c not in ("id", "is_fraud", "fold")]
    x_train_raw, x_val_raw = train_df[feature_cols], val_df[feature_cols]
    y_train = train_df["is_fraud"].to_numpy(dtype=np.float32, copy=True)
    y_val = val_df["is_fraud"].to_numpy(dtype=np.int32, copy=True)

    # ===== 3. Tiền xử lý: chỉ FIT trên train =====
    num_cols = x_train_raw.select_dtypes(include="number").columns.tolist()
    cat_cols = x_train_raw.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    bool_cols = x_train_raw.select_dtypes(include="bool").columns.tolist()
    assert len(num_cols) + len(cat_cols) + len(bool_cols) == len(feature_cols)

    preprocessor = ColumnTransformer([
        ("numbers", StandardScaler(), num_cols),
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("booleans", "passthrough", bool_cols),
    ])
    x_train = np.asarray(preprocessor.fit_transform(x_train_raw), dtype=np.float32).copy()
    x_val = np.asarray(preprocessor.transform(x_val_raw), dtype=np.float32).copy()
    assert np.isfinite(x_train).all() and np.isfinite(x_val).all(), "Dữ liệu chứa NaN/Inf"

    # ===== 4. Tạo MLP =====
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1),
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train).unsqueeze(1)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val)), batch_size=4096)

    # ===== 5. Huấn luyện và chọn epoch tốt nhất =====
    best_auc, best_epoch = -1.0, 0
    best_weights, best_predictions = None, None
    print(f"\nFOLD {fold} | train={len(train_df):,} | val={len(val_df):,} | "
          f"features={x_train.shape[1]} | device={DEVICE}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        model.eval()
        predictions = []
        with torch.no_grad():
            for (batch_x,) in val_loader:
                probs = torch.sigmoid(model(batch_x.to(DEVICE)))
                predictions.append(probs.cpu().numpy().ravel())
        predictions = np.concatenate(predictions)
        auc = roc_auc_score(y_val, predictions)
        print(f"  Epoch {epoch}/{EPOCHS} | loss={total_loss / len(train_df):.5f} | AUC={auc:.6f}")

        if auc > best_auc:
            best_auc, best_epoch = float(auc), epoch
            best_weights = copy.deepcopy(model.state_dict())
            best_predictions = predictions.copy()

    # ===== 6. Lưu mô hình và OOF của fold này =====
    model_dir = ROOT / "models"
    oof_dir = ROOT / "predictions" / "oof"
    model_dir.mkdir(parents=True, exist_ok=True)
    oof_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_weights, "input_dim": x_train.shape[1]},
               model_dir / f"mlp_fold{fold}.pt")
    joblib.dump(preprocessor, model_dir / f"mlp_fold{fold}_preprocessor.joblib")

    fold_oof = pd.DataFrame({
        "id": val_df["id"].to_numpy(),
        "fold": fold,
        "y_true": y_val,
        "oof_prediction": best_predictions,
    })
    fold_oof.to_csv(oof_dir / f"mlp_fold{fold}_oof.csv", index=False)
    return {"fold": fold, "best_epoch": best_epoch, "roc_auc": best_auc}, fold_oof


def main():
    # Đọc dữ liệu một lần, sau đó chạy lần lượt 5 folds.
    train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    folds = pd.read_csv(ROOT / "validation" / "folds.csv")
    assert train["id"].is_unique and folds["id"].is_unique, "ID trùng lặp"
    data = train.merge(folds[["id", "fold"]], on="id", validate="one_to_one", sort=False)
    assert len(data) == len(train) and set(data["fold"].unique()) == set(range(5)), "folds.csv không hợp lệ"
    assert not data.isna().any().any(), "Baseline này yêu cầu dữ liệu không có giá trị thiếu"

    scores, all_oof = [], []
    for fold in range(5):
        result, fold_oof = train_one_fold(data, fold)
        scores.append(result)
        all_oof.append(fold_oof)

    # ===== 7. Tổng hợp đủ 5 folds =====
    evaluation_dir = ROOT / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(evaluation_dir / "mlp_cv_results.csv", index=False)

    oof = pd.concat(all_oof, ignore_index=True).sort_values("id")
    assert len(oof) == len(train) and oof["id"].is_unique, "OOF thiếu hoặc trùng ID"
    oof.to_csv(ROOT / "predictions" / "oof" / "mlp_oof_full.csv", index=False)

    print("\n========== KẾT QUẢ 5 FOLDS ==========")
    print(scores_df.to_string(index=False))
    print(f"Mean ROC AUC: {scores_df['roc_auc'].mean():.6f}")
    print(f"Std ROC AUC : {scores_df['roc_auc'].std(ddof=1):.6f}")
    print(f"OOF ROC AUC : {roc_auc_score(oof['y_true'], oof['oof_prediction']):.6f}")
    print("Đã lưu OOF đủ 5 folds. Chưa tạo dự đoán test hoặc submission.")


if __name__ == "__main__":
    main()
