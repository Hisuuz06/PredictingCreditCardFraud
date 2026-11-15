from __future__ import annotations

from catboost import CatBoostClassifier


def build_model(
    *,
    iterations: int = 500,
    depth: int = 6,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> CatBoostClassifier:
    """Build a baseline; pass categorical column names to fit(..., cat_features=...)."""
    return CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )
