"""
Evaluation metrics for coral bleaching model assessment.

This module provides functions for computing classification metrics and
logging them to experiment tracking tools like Weights & Biases.
"""

from typing import Dict, Any, Optional, Union, List
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix as sk_confusion_matrix,
    classification_report,
)


def to_numpy(tensor_or_array: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Convert tensor or array to numpy array.

    Args:
        tensor_or_array: Input tensor or numpy array.

    Returns:
        np.ndarray: Numpy array.
    """
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().numpy()
    return np.asarray(tensor_or_array)


def compute_accuracy(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Compute classification accuracy.

    Args:
        y_true: True labels, shape (N,).
        y_pred: Predicted labels, shape (N,).

    Returns:
        float: Accuracy score between 0 and 1.

    Examples:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> compute_accuracy(y_true, y_pred)
        0.75
    """
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)

    return float(accuracy_score(y_true, y_pred))


def compute_confusion_matrix(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    normalize: Optional[str] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
        normalize: Normalization mode ('true', 'pred', 'all', or None).

    Returns:
        np.ndarray: Confusion matrix of shape (n_classes, n_classes).

    Examples:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> cm = compute_confusion_matrix(y_true, y_pred)
        >>> cm
        array([[2, 0],
               [1, 1]])
    """
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)

    return sk_confusion_matrix(y_true, y_pred, normalize=normalize)


def compute_precision(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    average: str = 'binary',
    **kwargs
) -> Union[float, np.ndarray]:
    """
    Compute precision score.

    Args:
        y_true: True labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
        average: Averaging mode ('binary', 'micro', 'macro', 'weighted', or None).
        **kwargs: Additional arguments for sklearn.metrics.precision_score.

    Returns:
        float or np.ndarray: Precision score(s).

    Examples:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> compute_precision(y_true, y_pred)
        1.0
    """
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)

    return precision_score(y_true, y_pred, average=average, zero_division=0, **kwargs)


def compute_recall(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    average: str = 'binary',
    **kwargs
) -> Union[float, np.ndarray]:
    """
    Compute recall score.

    Args:
        y_true: True labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
        average: Averaging mode ('binary', 'micro', 'macro', 'weighted', or None).
        **kwargs: Additional arguments for sklearn.metrics.recall_score.

    Returns:
        float or np.ndarray: Recall score(s).

    Examples:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> compute_recall(y_true, y_pred)
        0.5
    """
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)

    return recall_score(y_true, y_pred, average=average, zero_division=0, **kwargs)


def compute_f1(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    average: str = 'binary',
    **kwargs
) -> Union[float, np.ndarray]:
    """
    Compute F1 score.

    Args:
        y_true: True labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
        average: Averaging mode ('binary', 'micro', 'macro', 'weighted', or None).
        **kwargs: Additional arguments for sklearn.metrics.f1_score.

    Returns:
        float or np.ndarray: F1 score(s).

    Examples:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> compute_f1(y_true, y_pred)
        0.666...
    """
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)

    return f1_score(y_true, y_pred, average=average, zero_division=0, **kwargs)


def compute_classification_metrics(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Computes accuracy, precision, recall, F1 (both overall and per-class),
    and confusion matrix.

    Args:
        y_true: True labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
        class_names: Optional list of class names for per-class metrics.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - accuracy: Overall accuracy
            - precision_macro: Macro-averaged precision
            - recall_macro: Macro-averaged recall
            - f1_macro: Macro-averaged F1
            - precision_per_class: Precision for each class
            - recall_per_class: Recall for each class
            - f1_per_class: F1 for each class
            - confusion_matrix: Confusion matrix

    Examples:
        >>> y_true = np.array([0, 1, 1, 0, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0, 1, 1])
        >>> metrics = compute_classification_metrics(y_true, y_pred)
        >>> print(f"Accuracy: {metrics['accuracy']:.2f}")
        Accuracy: 0.67
    """
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)

    metrics = {
        # Overall metrics
        'accuracy': compute_accuracy(y_true, y_pred),
        'precision_macro': float(compute_precision(y_true, y_pred, average='macro')),
        'recall_macro': float(compute_recall(y_true, y_pred, average='macro')),
        'f1_macro': float(compute_f1(y_true, y_pred, average='macro')),

        # Per-class metrics
        'precision_per_class': compute_precision(y_true, y_pred, average=None).tolist(),
        'recall_per_class': compute_recall(y_true, y_pred, average=None).tolist(),
        'f1_per_class': compute_f1(y_true, y_pred, average=None).tolist(),

        # Confusion matrix
        'confusion_matrix': compute_confusion_matrix(y_true, y_pred).tolist(),
    }

    # Add class names if provided
    if class_names is not None:
        metrics['class_names'] = class_names

        # Create per-class dictionaries
        metrics['precision_by_class'] = {
            name: prec
            for name, prec in zip(class_names, metrics['precision_per_class'])
        }
        metrics['recall_by_class'] = {
            name: rec
            for name, rec in zip(class_names, metrics['recall_per_class'])
        }
        metrics['f1_by_class'] = {
            name: f1
            for name, f1 in zip(class_names, metrics['f1_per_class'])
        }

    return metrics


def print_classification_report(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Generate and return a text classification report.

    Args:
        y_true: True labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
        class_names: Optional list of class names.

    Returns:
        str: Classification report as string.

    Examples:
        >>> y_true = np.array([0, 1, 1, 0, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0, 1, 1])
        >>> report = print_classification_report(y_true, y_pred, ['bleached', 'healthy'])
        >>> print(report)
    """
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)

    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )


def log_metrics_to_wandb(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    prefix: str = "",
) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics: Dictionary of metrics to log.
        step: Optional step number for logging.
        prefix: Optional prefix for metric names (e.g., 'train/', 'val/').

    Examples:
        >>> metrics = {'accuracy': 0.95, 'loss': 0.1}
        >>> log_metrics_to_wandb(metrics, step=100, prefix='train/')
        # Logs as 'train/accuracy' and 'train/loss' to wandb
    """
    try:
        import wandb

        if wandb.run is None:
            # No active wandb run, skip logging
            return

        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        # Log to wandb
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)

    except ImportError:
        # wandb not installed, skip logging
        pass


def log_confusion_matrix_to_wandb(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    class_names: List[str],
    title: str = "Confusion Matrix",
) -> None:
    """
    Log confusion matrix to Weights & Biases.

    Args:
        y_true: True labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
        class_names: List of class names.
        title: Title for the confusion matrix plot.

    Examples:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> log_confusion_matrix_to_wandb(y_true, y_pred, ['bleached', 'healthy'])
    """
    try:
        import wandb

        if wandb.run is None:
            return

        # Compute confusion matrix
        cm = compute_confusion_matrix(y_true, y_pred)

        # Log as wandb plot
        wandb.log({
            title: wandb.plot.confusion_matrix(
                probs=None,
                y_true=to_numpy(y_true).tolist(),
                preds=to_numpy(y_pred).tolist(),
                class_names=class_names,
            )
        })

    except ImportError:
        pass


# Backwards compatibility
def compute_metrics(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute evaluation metrics.

    This is an alias for compute_classification_metrics() for backwards compatibility.

    Args:
        y_true: True labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
        class_names: Optional list of class names.

    Returns:
        Dict[str, Any]: Dictionary of metrics.
    """
    return compute_classification_metrics(y_true, y_pred, class_names)


if __name__ == "__main__":
    # Quick diagnostic when run directly
    print("=== Metrics Module Test ===\n")

    # Create sample data
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 0])

    class_names = ['bleached', 'healthy']

    print("Sample Data:")
    print(f"  y_true: {y_true}")
    print(f"  y_pred: {y_pred}")

    print("\n=== Individual Metrics ===")
    print(f"Accuracy: {compute_accuracy(y_true, y_pred):.3f}")
    print(f"Precision: {compute_precision(y_true, y_pred, average='macro'):.3f}")
    print(f"Recall: {compute_recall(y_true, y_pred, average='macro'):.3f}")
    print(f"F1 Score: {compute_f1(y_true, y_pred, average='macro'):.3f}")

    print("\n=== Confusion Matrix ===")
    cm = compute_confusion_matrix(y_true, y_pred)
    print(cm)

    print("\n=== Comprehensive Metrics ===")
    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    for key, value in metrics.items():
        if key not in ['confusion_matrix', 'class_names']:
            print(f"{key}: {value}")

    print("\n=== Classification Report ===")
    report = print_classification_report(y_true, y_pred, class_names)
    print(report)
