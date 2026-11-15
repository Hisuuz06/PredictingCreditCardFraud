from __future__ import annotations

from lightgbm import LGBMClassifier


def build_model(
    *,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    random_state: int = 42,
) -> LGBMClassifier:
    """Build a baseline; convert categorical columns to pandas category before fit."""
    return LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
