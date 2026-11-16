"""Kiểm tra nhanh pipeline: v1 vs v2, so sánh holdout AUC bằng LightGBM (nhanh, chỉ để định hướng)."""
import sys, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from features.feature_pipeline import FeaturePipeline
from features import feature_config as cfg

P = sys.argv[1] if len(sys.argv) > 1 else "data/raw/"
train = pd.read_csv(P + "train.csv"); test = pd.read_csv(P + "test.csv")
y = train[cfg.TARGET]
tr_idx, va_idx = train_test_split(train.index, test_size=0.2, stratify=y, random_state=42)

for v in ["v1", "v2"]:
    fp = FeaturePipeline(v).fit(train.loc[tr_idx])
    Xt, Xv = fp.transform(train.loc[tr_idx]), fp.transform(train.loc[va_idx])
    m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.1, num_leaves=63, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=1)
    m.fit(Xt, y[tr_idx], eval_set=[(Xv, y[va_idx])], eval_metric="auc",
          callbacks=[lgb.early_stopping(50, verbose=False)])
    print(v, "n_features", Xt.shape[1], "best_iter", m.best_iteration_,
          "holdout AUC", round(roc_auc_score(y[va_idx], m.predict_proba(Xv)[:, 1]), 5))
