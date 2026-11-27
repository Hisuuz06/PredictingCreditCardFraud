"""Train 5-fold MLP with the shared validation folds.

Run from the project root:
    .\\.venv\\Scripts\\python.exe -m dl.scripts.train_mlp --variant baseline
    .\\.venv\\Scripts\\python.exe -m dl.scripts.train_mlp --variant v3_linear --epochs 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dl.trainers.train_mlp import train_mlp


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 5-fold MLP model.")
    parser.add_argument(
        "--variant",
        default="baseline",
        choices=["baseline", "v3_linear"],
        help="Preprocessing variant to use.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_mlp(
        variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
