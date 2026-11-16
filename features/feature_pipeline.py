"""Feature pipeline - Người 1.
Dùng:
    from features.feature_pipeline import FeaturePipeline
    fp = FeaturePipeline(version="v2").fit(train_df)      # fit trên TRAIN (hoặc trên fold-train)
    X_train = fp.transform(train_df); X_test = fp.transform(test_df)
Ghi chú: v1/v2 chỉ dùng phép biến đổi theo từng dòng nên không có leakage giữa các fold.
Nếu sau này thêm thống kê theo nhóm/target encoding, BẮT BUỘC fit trong fold-train.
"""
import numpy as np
import pandas as pd
from . import feature_config as cfg


class FeaturePipeline:
    def __init__(self, version: str = cfg.FEATURE_VERSION, for_dl: bool = False):
        assert version in cfg.CHANGELOG, f"Version không tồn tại: {version}"
        self.version, self.for_dl = version, for_dl
        self.categories_ = {}

    def fit(self, df: pd.DataFrame):
        # Chỉ ghi nhớ bộ giá trị category để train/test dùng cùng mã
        self.categories_ = {c: sorted(df[c].dropna().unique()) for c in cfg.CAT_COLS}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.drop(columns=[c for c in (cfg.ID_COL, cfg.TARGET) if c in df.columns]).copy()

        for c in cfg.BOOL_COLS:
            out[c] = out[c].astype(int)

        if self.version == "v2":
            out["log_amount"] = np.log1p(out["amount_usd"])
            out["amount_to_balance"] = out["amount_usd"] / (out["account_balance_usd"] + 1.0)
            out["txn_per_hour"] = out["txn_count_last_24h"] / (out["hours_since_last_txn"] + 0.01)
            out["risk_flag_count"] = out[cfg.RISK_FLAGS].sum(axis=1)
            out["no_auth"] = (out["auth_method"] == "No Authentication").astype(int)
            out["is_night"] = out["time_of_day_hour"].between(0, 5).astype(int)
            out["velocity_x_merchant_risk"] = out["velocity_score"] * out["merchant_risk_score"] / 100.0
            if self.for_dl:   # DL thích biểu diễn tuần hoàn cho giờ
                out["hour_sin"] = np.sin(2 * np.pi * out["time_of_day_hour"] / 24)
                out["hour_cos"] = np.cos(2 * np.pi * out["time_of_day_hour"] / 24)

        for c, cats in self.categories_.items():
            out[c] = pd.Categorical(out[c], categories=cats)   # giá trị lạ -> NaN
        return out

    def feature_names(self, df: pd.DataFrame):
        return list(self.transform(df.head(5)).columns)
