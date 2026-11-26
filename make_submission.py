"""Tạo file nộp Kaggle từ predictions/test/<tên>.csv và kiểm tra định dạng.
Chạy từ thư mục gốc dự án:
    python make_submission.py                 # mặc định: lr_v3_linear
    python make_submission.py lr_v1           # chọn file khác trong predictions/test/
Kết quả: submission_<tên>.csv ngay trong thư mục hiện tại.
"""
import sys
import pandas as pd

name = sys.argv[1] if len(sys.argv) > 1 else "lr_v3_linear"

t = pd.read_csv(f"predictions/test/{name}.csv").rename(columns={"prediction": "is_fraud"})
ref = pd.read_csv("data/raw/sample_submission.csv")

assert list(t.columns) == ["id", "is_fraud"], f"Sai tên cột: {list(t.columns)}"
assert len(t) == len(ref), f"Sai số dòng: {len(t)} (cần {len(ref)})"
assert t["id"].equals(ref["id"]), "Cột id không khớp sample_submission.csv"
assert t["is_fraud"].between(0, 1).all(), "Có giá trị ngoài [0, 1]"
assert not t.isna().any().any(), "Có giá trị thiếu"

out = f"submission_{name}.csv"
t.to_csv(out, index=False)
print(f"OK: đã tạo {out} ({t.shape[0]} dòng). Upload file này lên Kaggle.")
