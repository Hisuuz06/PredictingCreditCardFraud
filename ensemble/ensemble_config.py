from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelPredictionSpec:
    name: str
    oof_path: Path
    test_path: Path


MODEL_SPECS: tuple[ModelPredictionSpec, ...] = (
    ModelPredictionSpec(
        name="catboost_v1",
        oof_path=ROOT / "predictions" / "oof" / "ml_v1" / "catboost_oof.csv",
        test_path=ROOT / "predictions" / "test" / "ml_v1" / "catboost_test.csv",
    ),
    ModelPredictionSpec(
        name="lightgbm_v1",
        oof_path=ROOT / "predictions" / "oof" / "ml_v1" / "lightgbm_oof.csv",
        test_path=ROOT / "predictions" / "test" / "ml_v1" / "lightgbm_test.csv",
    ),
    ModelPredictionSpec(
        name="xgboost_v1",
        oof_path=ROOT / "predictions" / "oof" / "ml_v1" / "xgboost_oof.csv",
        test_path=ROOT / "predictions" / "test" / "ml_v1" / "xgboost_test.csv",
    ),
    ModelPredictionSpec(
        name="lr_v3_linear",
        oof_path=ROOT / "predictions" / "oof" / "lr_v3_linear.csv",
        test_path=ROOT / "predictions" / "test" / "lr_v3_linear.csv",
    ),
    ModelPredictionSpec(
        name="mlp_v3_linear",
        oof_path=ROOT / "predictions" / "oof" / "dl_v3_linear" / "mlp_oof.csv",
        test_path=ROOT / "predictions" / "test" / "dl_v3_linear" / "mlp_test.csv",
    ),
)

SAMPLE_SUBMISSION_PATH = ROOT / "data" / "raw" / "sample_submission.csv"
ENSEMBLE_OOF_PATH = ROOT / "predictions" / "oof" / "ensemble_oof.csv"
ENSEMBLE_TEST_PATH = ROOT / "predictions" / "test" / "ensemble_test.csv"
FINAL_SUBMISSION_PATH = ROOT / "submissions" / "submission_final.csv"
ENSEMBLE_RESULTS_PATH = ROOT / "evaluation" / "ensemble_results.json"
