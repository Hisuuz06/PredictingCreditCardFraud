"""Evaluate OOF blends (weighted & rank averaging) and build submission_final.csv.

Run from the project root:
    .\\.venv\\Scripts\\python.exe -m ensemble.ensemble
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.ensemble_config import (
    ENSEMBLE_OOF_PATH,
    ENSEMBLE_RESULTS_PATH,
    ENSEMBLE_TEST_PATH,
    FINAL_SUBMISSION_PATH,
    MODEL_SPECS,
    SAMPLE_SUBMISSION_PATH,
    ModelPredictionSpec,
)


def _load_oof_matrix(
    specs: tuple[ModelPredictionSpec, ...],
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    base_ids: pd.Series | None = None
    base_folds: pd.Series | None = None
    base_y: pd.Series | None = None
    oof_cols: dict[str, np.ndarray] = {}

    for spec in specs:
        if not spec.oof_path.is_file():
            raise FileNotFoundError(f"Missing OOF file for {spec.name}: {spec.oof_path}")
        df = pd.read_csv(spec.oof_path)
        required = {"id", "fold", "y_true", "oof_prediction"}
        if not required.issubset(df.columns):
            raise ValueError(f"{spec.oof_path} missing required columns {required}")
        df = df.sort_values("id").reset_index(drop=True)
        if df["id"].duplicated().any():
            raise ValueError(f"Duplicate IDs found in {spec.oof_path}")

        preds = df["oof_prediction"].to_numpy(dtype=np.float64)
        if not np.isfinite(preds).all() or ((preds < 0) | (preds > 1)).any():
            raise ValueError(f"Invalid OOF probabilities in {spec.oof_path}")

        if base_ids is None:
            base_ids = df["id"]
            base_folds = df["fold"].astype(int)
            base_y = df["y_true"].astype(int)
        else:
            if not df["id"].equals(base_ids):
                raise ValueError(f"IDs in {spec.oof_path} do not match {specs[0].oof_path}")
            if not df["fold"].astype(int).equals(base_folds):
                raise ValueError(f"Folds in {spec.oof_path} do not match {specs[0].oof_path}")
            if not df["y_true"].astype(int).equals(base_y):
                raise ValueError(f"y_true in {spec.oof_path} does not match {specs[0].oof_path}")

        oof_cols[spec.name] = preds

    assert base_ids is not None and base_folds is not None and base_y is not None
    return base_ids, base_folds, base_y, pd.DataFrame(oof_cols)


def _load_test_matrix(
    specs: tuple[ModelPredictionSpec, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    if list(sample.columns) != ["id", "is_fraud"] or sample["id"].duplicated().any():
        raise ValueError("Invalid sample_submission.csv")

    test_cols: dict[str, np.ndarray] = {}
    for spec in specs:
        if not spec.test_path.is_file():
            raise FileNotFoundError(f"Missing test file for {spec.name}: {spec.test_path}")
        df = pd.read_csv(spec.test_path)
        if not {"id", "prediction"}.issubset(df.columns):
            raise ValueError(f"{spec.test_path} must contain id, prediction")
        if df["id"].duplicated().any() or set(df["id"]) != set(sample["id"]):
            raise ValueError(f"Test IDs mismatch in {spec.test_path}")

        aligned = df.set_index("id").loc[sample["id"], "prediction"].to_numpy(dtype=np.float64)
        if not np.isfinite(aligned).all() or ((aligned < 0) | (aligned > 1)).any():
            raise ValueError(f"Invalid test probabilities in {spec.test_path}")
        test_cols[spec.name] = aligned

    return sample, pd.DataFrame(test_cols)


def _to_rank_matrix(df: pd.DataFrame) -> np.ndarray:
    ranked = np.column_stack(
        [rankdata(df[col].to_numpy(), method="average") / len(df) for col in df.columns]
    )
    return ranked


def _fold_cv_stats(y_true: np.ndarray, folds: np.ndarray, preds: np.ndarray) -> tuple[float, float, float]:
    fold_aucs = [
        float(roc_auc_score(y_true[folds == f], preds[folds == f]))
        for f in sorted(np.unique(folds))
    ]
    return float(np.mean(fold_aucs)), float(np.std(fold_aucs, ddof=1)), float(roc_auc_score(y_true, preds))


def _optimize_weights(matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    n_models = matrix.shape[1]
    init_w = np.full(n_models, 1.0 / n_models, dtype=np.float64)

    # Use log-loss on probabilities or rank-transformed scores as a smooth surrogate
    # first, then refine with Powell directly on ROC AUC.
    eps = 1e-7
    clipped = np.clip(matrix, eps, 1.0 - eps)

    def neg_logloss(weights: np.ndarray) -> float:
        blend = np.clip(clipped @ weights, eps, 1.0 - eps)
        return float(-np.mean(y_true * np.log(blend) + (1 - y_true) * np.log(1 - blend)))

    slsqp = minimize(
        neg_logloss,
        init_w,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )
    start_w = slsqp.x if slsqp.success else init_w
    start_w = np.maximum(start_w, 0.0)
    start_w /= start_w.sum()

    def neg_auc(raw_w: np.ndarray) -> float:
        w = np.maximum(raw_w, 0.0)
        if w.sum() == 0:
            return 0.0
        w /= w.sum()
        return float(-roc_auc_score(y_true, matrix @ w))

    best_w = start_w
    best_auc = -neg_auc(best_w)
    for seed_w in (start_w, init_w):
        res = minimize(neg_auc, seed_w, method="Nelder-Mead", options={"maxiter": 400})
        cand = np.maximum(res.x, 0.0)
        if cand.sum() > 0:
            cand /= cand.sum()
            cand_auc = float(roc_auc_score(y_true, matrix @ cand))
            if cand_auc > best_auc:
                best_auc = cand_auc
                best_w = cand
    return best_w


def main() -> None:
    ids, folds_s, y_s, oof_df = _load_oof_matrix(MODEL_SPECS)
    sample, test_df = _load_test_matrix(MODEL_SPECS)
    y_true = y_s.to_numpy()
    folds = folds_s.to_numpy()
    model_names = list(oof_df.columns)

    print("=== INDIVIDUAL MODEL PERFORMANCE (5-FOLD) ===")
    individual_summary: dict[str, dict[str, float]] = {}
    for name in model_names:
        mean_auc, std_auc, oof_auc = _fold_cv_stats(y_true, folds, oof_df[name].to_numpy())
        individual_summary[name] = {
            "cv_mean": round(mean_auc, 6),
            "cv_std": round(std_auc, 6),
            "oof_auc": round(oof_auc, 6),
        }
        print(f"  {name:15s} | CV={mean_auc:.6f} ± {std_auc:.6f} | OOF={oof_auc:.6f}")

    print("\n=== OOF PREDICTION CORRELATION ===")
    print(oof_df.corr().round(4).to_string())

    oof_prob_mat = oof_df.to_numpy()
    test_prob_mat = test_df.to_numpy()
    oof_rank_mat = _to_rank_matrix(oof_df)
    test_rank_mat = _to_rank_matrix(test_df)

    equal_w = np.full(len(model_names), 1.0 / len(model_names))
    opt_prob_w = _optimize_weights(oof_prob_mat, y_true)
    opt_rank_w = _optimize_weights(oof_rank_mat, y_true)

    candidates = [
        ("equal_prob_avg", equal_w, oof_prob_mat @ equal_w, test_prob_mat @ equal_w),
        ("equal_rank_avg", equal_w, oof_rank_mat @ equal_w, test_rank_mat @ equal_w),
        ("weighted_prob_avg", opt_prob_w, oof_prob_mat @ opt_prob_w, test_prob_mat @ opt_prob_w),
        ("weighted_rank_avg", opt_rank_w, oof_rank_mat @ opt_rank_w, test_rank_mat @ opt_rank_w),
    ]

    print("\n=== ENSEMBLE CANDIDATES (OOF CV) ===")
    best_name = ""
    best_score = -1.0
    best_weights = equal_w
    best_oof = np.zeros_like(y_true, dtype=np.float64)
    best_test = np.zeros(len(sample), dtype=np.float64)
    candidate_summary: dict[str, dict[str, object]] = {}

    for method_name, weights, oof_blend, test_blend in candidates:
        mean_auc, std_auc, oof_auc = _fold_cv_stats(y_true, folds, oof_blend)
        weight_map = {m: round(float(w), 4) for m, w in zip(model_names, weights)}
        candidate_summary[method_name] = {
            "cv_mean": round(mean_auc, 6),
            "cv_std": round(std_auc, 6),
            "oof_auc": round(oof_auc, 6),
            "weights": weight_map,
        }
        print(f"  {method_name:18s} | CV={mean_auc:.6f} ± {std_auc:.6f} | OOF={oof_auc:.6f} | w={weight_map}")
        if oof_auc > best_score:
            best_score = oof_auc
            best_name = method_name
            best_weights = weights
            best_oof = oof_blend
            best_test = test_blend

    best_test = np.clip(best_test, 0.0, 1.0)
    if not np.isfinite(best_test).all() or ((best_test < 0) | (best_test > 1)).any():
        raise ValueError("Final ensemble test predictions contain invalid values.")

    ENSEMBLE_OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENSEMBLE_TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENSEMBLE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {"id": ids, "fold": folds, "y_true": y_true, "oof_prediction": best_oof}
    ).to_csv(ENSEMBLE_OOF_PATH, index=False)
    pd.DataFrame({"id": sample["id"], "prediction": best_test}).to_csv(
        ENSEMBLE_TEST_PATH, index=False
    )

    submission = sample.copy()
    submission["is_fraud"] = best_test
    assert list(submission.columns) == ["id", "is_fraud"]
    assert len(submission) == len(sample)
    assert submission["id"].equals(sample["id"])
    assert not submission.isna().any().any()
    submission.to_csv(FINAL_SUBMISSION_PATH, index=False)

    results_payload = {
        "individual_models": individual_summary,
        "ensemble_candidates": candidate_summary,
        "selected_method": best_name,
        "selected_weights": {m: round(float(w), 4) for m, w in zip(model_names, best_weights)},
        "selected_oof_auc": round(best_score, 6),
    }
    with open(ENSEMBLE_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)

    print(f"\nSelected: {best_name} | OOF AUC={best_score:.6f}")
    print(f"Saved OOF       : {ENSEMBLE_OOF_PATH}")
    print(f"Saved Test      : {ENSEMBLE_TEST_PATH}")
    print(f"Saved Submission: {FINAL_SUBMISSION_PATH} ({len(submission):,} rows)")


if __name__ == "__main__":
    main()
