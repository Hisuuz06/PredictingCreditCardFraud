from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from features.linear_preprocessor import make_linear_preprocessor


def build_mlp_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    boolean_columns: list[str],
    *,
    variant: str = "baseline",
) -> ColumnTransformer:
    """Build fold-safe preprocessing for MLP (`baseline` or `v3_linear`)."""
    if variant == "v3_linear":
        return make_linear_preprocessor(use_spline=True)
    if variant != "baseline":
        raise ValueError(f"Unsupported MLP variant: {variant}")

    return ColumnTransformer(
        transformers=[
            ("numbers", StandardScaler(), numeric_columns),
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
            ("booleans", "passthrough", boolean_columns),
        ],
        remainder="drop",
    )


def build_mlp_model(
    input_dim: int,
    *,
    hidden_dims: tuple[int, ...] = (64, 32),
    dropout: float = 0.2,
) -> nn.Module:
    """Build tabular MLP binary classifier outputting logits."""
    layers: list[nn.Module] = []
    in_dim = input_dim
    for idx, hidden_dim in enumerate(hidden_dims):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        if idx == 0 and dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)
