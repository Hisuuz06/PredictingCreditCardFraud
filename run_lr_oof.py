"""Chạy Logistic Regression 5-fold, so sánh v1 (tuyến tính) với v3_linear (spline), lưu OOF đúng định dạng kế hoạch.
Chạy:  python run_lr_oof.py data/raw/
Tuỳ chọn: --sample 100000 (chạy thử nhanh) --folds 5
Folds: nếu có validation/folds.csv (cột id, fold) thì dùng, để cả nhóm chung fold.
Kết quả: predictions/oof/lr_<variant>.csv (id, fold, y_true, oof_prediction)
         predictions/test/lr_<variant>.csv (id, prediction = trung bình các fold)
         predictions/meta/lr_<variant>.json (feature_version, seed, tham số, CV mean/std, OOF AUC)
"""
import os, json, argparse, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from features.feature_pipeline import FeaturePipeline
from features.linear_preprocessor import make_linear_preprocessor
from features import feature_config as cfg
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("data_dir", nargs="?", default="data/raw/")
ap.add_argument("--sample", type=int, default=0)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--solver", default="lbfgs", choices=["lbfgs", "liblinear"])          # lbfgs nhanh hơn nhiều khi có spline
ap.add_argument("--class-weight", default="none", choices=["none", "balanced"])       # nhóm đang dùng balanced; AUC gần như không đổi
args = ap.parse_args()
SEED = 42
LR_PARAMS = dict(C=1.0, class_weight=None if args.class_weight == "none" else "balanced", max_iter=1000,
                 penalty="l2", solver=args.solver, random_state=SEED)

train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
if args.sample:
    train = train.sample(args.sample, random_state=0).reset_index(drop=True)
y = train[cfg.TARGET].values

if os.path.exists("validation/folds.csv") and not args.sample:
    fold = train[[cfg.ID_COL]].merge(pd.read_csv("validation/folds.csv"), on=cfg.ID_COL, how="left")["fold"].values
    print("Dùng validation/folds.csv")
else:
    fold = np.zeros(len(train), dtype=int)
    for k, (_, va) in enumerate(StratifiedKFold(args.folds, shuffle=True, random_state=SEED).split(train, y)):
        fold[va] = k
    print(f"Tự tạo {args.folds} folds (seed {SEED})")

fp = FeaturePipeline("v1").fit(train)
X, X_test = fp.transform(train), fp.transform(test)
for d in ("oof", "test", "meta"):
    os.makedirs(f"predictions/{d}", exist_ok=True)

results = {}
for variant, use_spline in [("v1", False), ("v3_linear", True)]:
    oof = np.zeros(len(X)); pred = np.zeros(len(X_test)); aucs = []
    folds_ids = sorted(np.unique(fold))
    for k in folds_ids:
        tr, va = fold != k, fold == k
        model = make_pipeline(make_linear_preprocessor(use_spline), LogisticRegression(**LR_PARAMS))
        model.fit(X[tr], y[tr])                      # preprocessor học CHỈ trên fold-train
        oof[va] = model.predict_proba(X[va])[:, 1]
        pred += model.predict_proba(X_test)[:, 1] / len(folds_ids)
        aucs.append(roc_auc_score(y[va], oof[va]))
        print(f"  [{variant}] fold {k}: AUC={aucs[-1]:.5f}", flush=True)
    name = f"lr_{variant}"
    pd.DataFrame({"id": train[cfg.ID_COL], "fold": fold, "y_true": y, "oof_prediction": oof}).to_csv(f"predictions/oof/{name}.csv", index=False)
    pd.DataFrame({"id": test[cfg.ID_COL], "prediction": pred}).to_csv(f"predictions/test/{name}.csv", index=False)
    meta = dict(model="LogisticRegression", feature_version=variant, seed=SEED, params=LR_PARAMS,
                preprocessing="StandardScaler + OneHot" + (" + Spline(8 knots) cho " + ",".join(cfg.SPLINE_COLS) if use_spline else ""),
                cv_auc_mean=round(float(np.mean(aucs)), 6), cv_auc_std=round(float(np.std(aucs)), 6),
                oof_auc=round(float(roc_auc_score(y, oof)), 6), n_folds=len(folds_ids), sampled=bool(args.sample))
    json.dump(meta, open(f"predictions/meta/{name}.json", "w"), ensure_ascii=False, indent=2)
    results[variant] = np.array(aucs)
    print(f"{name}: CV mean {meta['cv_auc_mean']} ± {meta['cv_auc_std']} | OOF AUC {meta['oof_auc']}\n", flush=True)

d = results["v3_linear"] - results["v1"]
print("Delta v3_linear - v1 theo fold:", np.round(d, 5), "| trung bình", round(d.mean(), 5))
print("Giữ v3_linear cho LR/MLP nếu delta > 0 ở (gần như) mọi fold.")
