from __future__ import annotations

from dl.trainers.cv_training import train_mlp_cross_validation


def train_mlp(
    *,
    variant: str = "baseline",
    epochs: int = 5,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    hidden_dims: tuple[int, ...] = (64, 32),
    dropout: float = 0.2,
    seed: int = 42,
) -> None:
    """Train 5-fold MLP model for the requested feature variant."""
    train_mlp_cross_validation(
        variant=variant,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_dims=hidden_dims,
        dropout=dropout,
        seed=seed,
    )
