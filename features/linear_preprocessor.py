"""Tiền xử lý v3_linear – CHỈ cho Logistic Regression / MLP (cây không cần).
Đầu vào: bảng đã qua FeaturePipeline("v1").transform (bool đã là 0/1, cột chữ là category).
  - 5 cột SPLINE_COLS  -> SplineTransformer(8 knots, bậc 3) rồi chuẩn hoá
  - cột số còn lại     -> chuẩn hoá
  - cột chữ            -> one-hot
  - cột bool           -> giữ nguyên
Spline và scaler HỌC từ dữ liệu => phải fit TRONG fold-train (đặt vào sklearn Pipeline là tự đúng).
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from . import feature_config as cfg


def make_linear_preprocessor(use_spline: bool = True) -> ColumnTransformer:
    spline_cols = cfg.SPLINE_COLS if use_spline else []
    other_num = [c for c in cfg.NUM_COLS if c not in spline_cols]
    parts = [
        ("num", StandardScaler(), other_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cfg.CAT_COLS),
        ("bool", "passthrough", cfg.BOOL_COLS),
    ]
    if spline_cols:
        parts.append(("spl", make_pipeline(SplineTransformer(n_knots=8, degree=3), StandardScaler()), spline_cols))
    return ColumnTransformer(parts)
