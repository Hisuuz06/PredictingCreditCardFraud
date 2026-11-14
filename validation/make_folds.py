"""Tạo Stratified 5-Fold cố định, dùng chung cho ML/DL."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]


def main():
    train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    assert train["id"].is_unique, "ID train không duy nhất"
    y = train["is_fraud"].astype(int)
    assert not y.isna().any() and set(y.unique()) <= {0, 1}

    fold_ids = np.full(len(train), -1, dtype=np.int8)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, val_idx) in enumerate(splitter.split(train, y)):
        fold_ids[val_idx] = fold
    assert (fold_ids >= 0).all()

    result = pd.DataFrame({"id": train["id"], "fold": fold_ids})
    output = ROOT / "validation" / "folds.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Đã tạo: {output}")
    stats = pd.DataFrame({"fold": fold_ids, "is_fraud": y}).groupby("fold")["is_fraud"].agg(["count", "sum", "mean"])
    print(stats.rename(columns={"count": "so_dong", "sum": "so_gian_lan", "mean": "ty_le_gian_lan"}).round(5))
    print("Mỗi ID thuộc đúng một fold; các mô hình cần dùng chính file này.")


if __name__ == "__main__":
    main()
