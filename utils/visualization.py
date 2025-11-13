"""
Visualization helpers for coral bleaching experiments.

This module provides plotting functions for training curves, confusion matrices,
and sample image grids. All functions use non-interactive matplotlib backend
for server/notebook compatibility.
"""

from typing import Optional, List, Union, Tuple
from pathlib import Path
import numpy as np
import torch

# Set non-interactive backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


def denormalize_image(
    tensor: Union[torch.Tensor, np.ndarray],
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> np.ndarray:
    """
    Denormalize image tensor for visualization.

    Reverses ImageNet normalization to get image back to [0, 1] range.

    Args:
        tensor: Image tensor of shape (C, H, W) or (H, W, C).
        mean: Mean used for normalization (per channel).
        std: Std used for normalization (per channel).

    Returns:
        np.ndarray: Denormalized image in [0, 1] range, shape (H, W, C).

    Examples:
        >>> tensor = torch.randn(3, 224, 224)  # Normalized
        >>> img = denormalize_image(tensor)
        >>> img.shape
        (224, 224, 3)
    """
    # Convert to numpy if tensor
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    # Handle both (C, H, W) and (H, W, C)
    if tensor.shape[0] == 3 or tensor.shape[0] == 1:
        # (C, H, W) -> (H, W, C)
        tensor = np.transpose(tensor, (1, 2, 0))

    # Denormalize
    mean = np.array(mean).reshape(1, 1, -1)
    std = np.array(std).reshape(1, 1, -1)

    img = tensor * std + mean

    # Clip to [0, 1]
    img = np.clip(img, 0, 1)

    return img


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot training and validation curves.

    Creates a figure with loss curves and optionally accuracy curves.

    Args:
        train_losses: Training losses per epoch.
        val_losses: Validation losses per epoch.
        train_accs: Optional training accuracies per epoch.
        val_accs: Optional validation accuracies per epoch.
        save_path: Optional path to save figure.

    Returns:
        plt.Figure: Matplotlib figure object.

    Examples:
        >>> train_losses = [0.5, 0.4, 0.3]
        >>> val_losses = [0.6, 0.5, 0.45]
        >>> fig = plot_training_curves(train_losses, val_losses)
    """
    # Determine number of subplots
    has_acc = train_accs is not None and val_accs is not None
    n_plots = 2 if has_acc else 1

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Plot loss
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy if provided
    if has_acc:
        axes[1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[int]]],
    class_names: List[str],
    normalize: bool = False,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.

    Args:
        cm: Confusion matrix of shape (n_classes, n_classes).
        class_names: List of class names.
        normalize: Whether to normalize by row (show percentages).
        save_path: Optional path to save figure.

    Returns:
        plt.Figure: Matplotlib figure object.

    Examples:
        >>> cm = np.array([[50, 10], [5, 35]])
        >>> fig = plot_confusion_matrix(cm, ['bleached', 'healthy'])
    """
    cm = np.array(cm)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2%' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage' if normalize else 'Count'},
        ax=ax
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_sample_grid(
    images: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray, List[int]],
    predictions: Optional[Union[torch.Tensor, np.ndarray, List[int]]] = None,
    class_names: Optional[List[str]] = None,
    num_samples: int = 16,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot grid of sample images with labels and optionally predictions.

    Args:
        images: Images tensor of shape (N, C, H, W) or (N, H, W, C).
        labels: Ground truth labels of shape (N,).
        predictions: Optional predictions of shape (N,).
        class_names: Optional class names for labels.
        num_samples: Number of samples to display.
        save_path: Optional path to save figure.

    Returns:
        plt.Figure: Matplotlib figure object.

    Examples:
        >>> images = torch.randn(16, 3, 224, 224)
        >>> labels = torch.randint(0, 2, (16,))
        >>> fig = plot_sample_grid(images, labels, class_names=['bleached', 'healthy'])
    """
    # Convert to numpy
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)

    if predictions is not None:
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        elif isinstance(predictions, list):
            predictions = np.array(predictions)

    # Limit to num_samples
    num_samples = min(num_samples, len(images))
    images = images[:num_samples]
    labels = labels[:num_samples]
    if predictions is not None:
        predictions = predictions[:num_samples]

    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for idx in range(grid_size * grid_size):
        ax = axes[idx]

        if idx < num_samples:
            # Denormalize image for display
            img = denormalize_image(images[idx])

            ax.imshow(img)

            # Create title
            label = labels[idx]
            label_str = class_names[label] if class_names else str(label)

            if predictions is not None:
                pred = predictions[idx]
                pred_str = class_names[pred] if class_names else str(pred)
                correct = (pred == label)
                color = 'green' if correct else 'red'
                title = f'True: {label_str}\nPred: {pred_str}'
                ax.set_title(title, fontsize=10, color=color, fontweight='bold')
            else:
                ax.set_title(f'Label: {label_str}', fontsize=10)

            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_class_distribution(
    label_counts: dict,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot class distribution as bar chart.

    Args:
        label_counts: Dictionary mapping class indices/names to counts.
        class_names: Optional class names for x-axis labels.
        save_path: Optional path to save figure.

    Returns:
        plt.Figure: Matplotlib figure object.

    Examples:
        >>> counts = {'bleached': 485, 'healthy': 438}
        >>> fig = plot_class_distribution(counts)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if class_names:
        labels = class_names
        counts = [label_counts.get(name, 0) for name in class_names]
    else:
        labels = list(label_counts.keys())
        counts = list(label_counts.values())

    bars = ax.bar(labels, counts, color=['skyblue', 'coral'][:len(labels)])

    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# Backwards compatibility
def plot_results(*args, **kwargs):
    """
    Backwards compatibility wrapper.

    Use specific plotting functions instead:
    - plot_training_curves()
    - plot_confusion_matrix()
    - plot_sample_grid()
    - plot_class_distribution()
    """
    raise NotImplementedError(
        "plot_results is deprecated. Use specific plotting functions:\n"
        "  - plot_training_curves()\n"
        "  - plot_confusion_matrix()\n"
        "  - plot_sample_grid()\n"
        "  - plot_class_distribution()"
    )


if __name__ == "__main__":
    # Quick diagnostic when run directly
    print("=== Visualization Module Test ===\n")

    # Test training curves
    train_losses = [0.8, 0.6, 0.4, 0.3, 0.25]
    val_losses = [0.9, 0.7, 0.55, 0.5, 0.48]
    train_accs = [0.6, 0.7, 0.8, 0.85, 0.88]
    val_accs = [0.55, 0.65, 0.72, 0.75, 0.76]

    print("Plotting training curves...")
    fig1 = plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    print("✓ Training curves created")
    plt.close(fig1)

    # Test confusion matrix
    cm = np.array([[45, 5], [10, 40]])
    class_names = ['bleached', 'healthy']

    print("Plotting confusion matrix...")
    fig2 = plot_confusion_matrix(cm, class_names)
    print("✓ Confusion matrix created")
    plt.close(fig2)

    # Test sample grid
    dummy_images = np.random.rand(16, 3, 64, 64)
    dummy_labels = np.random.randint(0, 2, 16)

    print("Plotting sample grid...")
    fig3 = plot_sample_grid(dummy_images, dummy_labels, class_names=class_names, num_samples=9)
    print("✓ Sample grid created")
    plt.close(fig3)

    # Test class distribution
    counts = {'bleached': 485, 'healthy': 438}

    print("Plotting class distribution...")
    fig4 = plot_class_distribution(counts)
    print("✓ Class distribution created")
    plt.close(fig4)

    print("\nAll visualization functions working!")
