from __future__ import annotations

from xgboost import XGBClassifier


def build_model(
    *,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    scale_pos_weight: float = 1.0,
    random_state: int = 42,
) -> XGBClassifier:
    """Build a baseline; categorical columns must use pandas category dtype."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        enable_categorical=True,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
    )
