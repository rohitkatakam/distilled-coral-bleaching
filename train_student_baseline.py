"""CLI entry point to train the baseline student model without knowledge distillation."""

from __future__ import annotations

import argparse

from models.student import StudentModel  # noqa: F401  # pylint: disable=unused-import
from utils.data_loader import build_dataloaders  # noqa: F401  # pylint: disable=unused-import


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for student baseline training."""
    parser = argparse.ArgumentParser(description="Train the student model baseline.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to the training config.")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to store student checkpoints.")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume from.")
    return parser.parse_args()


def main() -> None:
    """Execute baseline student training workflow."""
    _args = parse_args()
    raise NotImplementedError("Student baseline training loop is not yet implemented.")


if __name__ == "__main__":
    main()
