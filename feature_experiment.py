"""Thử TỪNG feature một bằng 5-fold CV, ghi kết quả vào evaluation/experiment_log.csv.
Chạy:  python feature_experiment.py data/raw/
Tuỳ chọn:  --sample 100000 (chạy thử nhanh)  --folds 5
Nếu có validation/folds.csv (cột id, fold) do TV2 tạo thì dùng, để cả nhóm chung folds.
"""
import os, sys, time, argparse, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from features.feature_pipeline import FeaturePipeline
from features import feature_config as cfg
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("data_dir", nargs="?", default="data/raw/")
ap.add_argument("--sample", type=int, default=0)
ap.add_argument("--folds", type=int, default=5)
args = ap.parse_args()

train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
if args.sample:
    train = train.sample(args.sample, random_state=0).reset_index(drop=True)
y = train[cfg.TARGET].values

# --- folds: ưu tiên file chung của nhóm ---
if os.path.exists("validation/folds.csv") and not args.sample:
    fold = train[[cfg.ID_COL]].merge(pd.read_csv("validation/folds.csv"), on=cfg.ID_COL, how="left")["fold"].values
    print("Dùng validation/folds.csv")
else:
    fold = np.zeros(len(train), dtype=int)
    for k, (_, va) in enumerate(StratifiedKFold(args.folds, shuffle=True, random_state=42).split(train, y)):
        fold[va] = k
    print(f"Tự tạo {args.folds} folds (seed 42)")

# --- ứng viên feature (mỗi cái chỉ dùng phép biến đổi theo từng dòng => không leakage) ---
CANDIDATES = {
    "log_amount":        lambda d: np.log1p(d["amount_usd"]),
    "amount_to_balance": lambda d: d["amount_usd"] / (d["account_balance_usd"] + 1),
    "txn_per_hour":      lambda d: d["txn_count_last_24h"] / (d["hours_since_last_txn"] + 0.01),
    "risk_flag_count":   lambda d: d[cfg.RISK_FLAGS].astype(int).sum(axis=1),
    "no_auth":           lambda d: (d["auth_method"] == "No Authentication").astype(int),
    "is_night":          lambda d: d["time_of_day_hour"].between(0, 5).astype(int),
    "velocity_x_risk":   lambda d: d["velocity_score"] * d["merchant_risk_score"] / 100,
}

base = FeaturePipeline("v1").fit(train).transform(train)

def cv_auc(X):
    scores = []
    for k in range(fold.max() + 1):
        tr, va = fold != k, fold == k
        m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.1, num_leaves=63, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=42)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False)])
        scores.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
    return np.mean(scores), np.std(scores)

rows = []
def run(name, X):
    t = time.time(); mean, std = cv_auc(X)
    rows.append(dict(feature_set=name, cv_auc_mean=round(mean, 5), cv_auc_std=round(std, 5),
                     n_features=X.shape[1], seconds=int(time.time() - t)))
    print(rows[-1], flush=True)

run("v1_baseline", base)
for name, fn in CANDIDATES.items():
    X = base.copy(); X[name] = fn(train)
    run("v1+" + name, X)

res = pd.DataFrame(rows)
res["delta_vs_baseline"] = (res.cv_auc_mean - res.cv_auc_mean.iloc[0]).round(5)
print("\n", res.to_string(index=False))
os.makedirs("evaluation", exist_ok=True)
res.to_csv("evaluation/experiment_log.csv", mode="a", index=False,
           header=not os.path.exists("evaluation/experiment_log.csv"))
print("\nĐã ghi evaluation/experiment_log.csv. Giữ feature nào có delta > 0 VÀ lớn hơn cv_auc_std.")
