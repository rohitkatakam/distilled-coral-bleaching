"""CLI entry point to evaluate trained teacher or student models on coral datasets."""

from __future__ import annotations

import argparse

from models.student import StudentModel  # noqa: F401  # pylint: disable=unused-import
from models.teacher import TeacherModel  # noqa: F401  # pylint: disable=unused-import
from utils.data_loader import build_dataloaders  # noqa: F401  # pylint: disable=unused-import
from utils.metrics import compute_metrics  # noqa: F401  # pylint: disable=unused-import


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model on coral bleaching datasets.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to the evaluation config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint containing model weights.")
    parser.add_argument("--model-type", type=str, choices=("teacher", "student"), default="student", help="Which model variant to evaluate.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate against.")
    return parser.parse_args()


def main() -> None:
    """Execute evaluation workflow."""
    _args = parse_args()
    raise NotImplementedError("Evaluation pipeline is not yet implemented.")


if __name__ == "__main__":
    main()
