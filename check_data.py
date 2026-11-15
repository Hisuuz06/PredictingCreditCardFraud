"""Kiểm tra train/test/sample_submission trước khi tạo folds và huấn luyện."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"


def main():
    train = pd.read_csv(RAW / "train.csv")
    test = pd.read_csv(RAW / "test.csv")
    sample = pd.read_csv(RAW / "sample_submission.csv")

    print("Train:", train.shape)
    print("Test:", test.shape)
    print("Sample submission:", sample.shape)

    assert {"id", "is_fraud"}.issubset(train.columns), "Train thiếu id hoặc is_fraud"
    assert "id" in test.columns, "Test thiếu id"
    assert {"id", "is_fraud"}.issubset(sample.columns), "Sample submission không có id/is_fraud"
    assert train["id"].is_unique and test["id"].is_unique, "Có ID bị trùng trong train/test"
    assert sample["id"].is_unique, "Có ID bị trùng trong sample submission"
    assert len(test) == len(sample), "Số dòng test và sample submission khác nhau"
    assert set(test["id"]) == set(sample["id"]), "Bộ ID test và sample submission khác nhau"

    feat_train = [c for c in train if c not in ("id", "is_fraud")]
    feat_test = [c for c in test if c != "id"]
    assert set(feat_train) == set(feat_test), "Các cột feature của train/test không khớp"
    assert not train["is_fraud"].isna().any(), "Train có nhãn thiếu"
    assert train["is_fraud"].isin([0, 1]).all(), "Target không thuần 0/1"

    print("\nSố giá trị thiếu train/test:")
    print("train =", int(train.isna().sum().sum()), "| test =", int(test.isna().sum().sum()))
    print("Trùng toàn bộ dòng train:", int(train.duplicated().sum()))
    print("\nPhân bố nhãn (%):")
    print((train["is_fraud"].value_counts(normalize=True).sort_index() * 100).round(4))
    print("\nCác cột categorical:")
    cat_cols = train[feat_train].select_dtypes(include=["object", "string", "category"]).columns
    for col in cat_cols:
        unseen = set(test[col].dropna().unique()) - set(train[col].dropna().unique())
        print(f"- {col}: {train[col].nunique()} giá trị train; {len(unseen)} giá trị chỉ có ở test")
    print("\nOK: Schema cơ bản hợp lệ. Nếu có missing test, pipeline phải xử lý trong từng fold.")


if __name__ == "__main__":
    main()
