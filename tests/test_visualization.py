"""
Tests for utils/visualization.py

Tests plotting functions with real matplotlib (not mocked).
"""

import pytest
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Ensure non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile

from utils.visualization import (
    denormalize_image,
    plot_training_curves,
    plot_confusion_matrix,
    plot_sample_grid,
    plot_class_distribution,
    plot_results,
)


class TestDenormalizeImage:
    """Tests for denormalize_image helper."""

    def test_denormalize_torch_tensor(self):
        """Test denormalization of torch tensor."""
        # Create normalized tensor
        tensor = torch.randn(3, 64, 64)

        result = denormalize_image(tensor)

        assert isinstance(result, np.ndarray)
        assert result.shape == (64, 64, 3)  # H, W, C
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_denormalize_numpy_array(self):
        """Test denormalization of numpy array."""
        array = np.random.randn(3, 64, 64)

        result = denormalize_image(array)

        assert isinstance(result, np.ndarray)
        assert result.shape == (64, 64, 3)

    def test_denormalize_hwc_format(self):
        """Test with H, W, C format input."""
        array = np.random.randn(64, 64, 3)

        result = denormalize_image(array)

        assert result.shape == (64, 64, 3)


class TestPlotTrainingCurves:
    """Tests for plot_training_curves function."""

    def test_returns_figure(self):
        """Test that function returns matplotlib Figure."""
        train_losses = [0.8, 0.6, 0.4]
        val_losses = [0.9, 0.7, 0.5]

        fig = plot_training_curves(train_losses, val_losses)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_accuracies(self):
        """Test plotting with accuracies."""
        train_losses = [0.8, 0.6, 0.4]
        val_losses = [0.9, 0.7, 0.5]
        train_accs = [0.6, 0.7, 0.8]
        val_accs = [0.55, 0.65, 0.75]

        fig = plot_training_curves(train_losses, val_losses, train_accs, val_accs)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Two subplots
        plt.close(fig)

    def test_without_accuracies(self):
        """Test plotting without accuracies."""
        train_losses = [0.8, 0.6, 0.4]
        val_losses = [0.9, 0.7, 0.5]

        fig = plot_training_curves(train_losses, val_losses)

        assert len(fig.axes) == 1  # One subplot
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        """Test that figure can be saved to file."""
        train_losses = [0.8, 0.6, 0.4]
        val_losses = [0.9, 0.7, 0.5]
        save_path = tmp_path / "curves.png"

        fig = plot_training_curves(train_losses, val_losses, save_path=save_path)

        assert save_path.exists()
        assert save_path.stat().st_size > 0
        plt.close(fig)


class TestPlotConfusionMatrix:
    """Tests for plot_confusion_matrix function."""

    def test_returns_figure(self):
        """Test that function returns matplotlib Figure."""
        cm = np.array([[45, 5], [10, 40]])
        class_names = ['bleached', 'healthy']

        fig = plot_confusion_matrix(cm, class_names)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_normalization(self):
        """Test with normalization enabled."""
        cm = np.array([[45, 5], [10, 40]])
        class_names = ['bleached', 'healthy']

        fig = plot_confusion_matrix(cm, class_names, normalize=True)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_list_input(self):
        """Test with list input instead of numpy array."""
        cm = [[45, 5], [10, 40]]
        class_names = ['bleached', 'healthy']

        fig = plot_confusion_matrix(cm, class_names)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        """Test saving confusion matrix to file."""
        cm = np.array([[45, 5], [10, 40]])
        class_names = ['bleached', 'healthy']
        save_path = tmp_path / "cm.png"

        fig = plot_confusion_matrix(cm, class_names, save_path=save_path)

        assert save_path.exists()
        plt.close(fig)


class TestPlotSampleGrid:
    """Tests for plot_sample_grid function."""

    def test_returns_figure(self):
        """Test that function returns matplotlib Figure."""
        images = torch.randn(16, 3, 64, 64)
        labels = torch.randint(0, 2, (16,))

        fig = plot_sample_grid(images, labels, num_samples=9)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_predictions(self):
        """Test with predictions provided."""
        images = torch.randn(16, 3, 64, 64)
        labels = torch.randint(0, 2, (16,))
        predictions = torch.randint(0, 2, (16,))

        fig = plot_sample_grid(images, labels, predictions, num_samples=9)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_class_names(self):
        """Test with class names."""
        images = torch.randn(16, 3, 64, 64)
        labels = torch.randint(0, 2, (16,))
        class_names = ['bleached', 'healthy']

        fig = plot_sample_grid(images, labels, class_names=class_names, num_samples=9)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_numpy_input(self):
        """Test with numpy arrays."""
        images = np.random.randn(16, 3, 64, 64)
        labels = np.random.randint(0, 2, 16)

        fig = plot_sample_grid(images, labels, num_samples=9)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotClassDistribution:
    """Tests for plot_class_distribution function."""

    def test_returns_figure(self):
        """Test that function returns matplotlib Figure."""
        counts = {'bleached': 485, 'healthy': 438}

        fig = plot_class_distribution(counts)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_class_names(self):
        """Test with class names provided."""
        counts = {0: 485, 1: 438}
        class_names = ['bleached', 'healthy']

        fig = plot_class_distribution(counts, class_names=class_names)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        """Test saving to file."""
        counts = {'bleached': 485, 'healthy': 438}
        save_path = tmp_path / "dist.png"

        fig = plot_class_distribution(counts, save_path=save_path)

        assert save_path.exists()
        plt.close(fig)


class TestPlotResults:
    """Tests for backwards compatibility function."""

    def test_raises_not_implemented(self):
        """Test that plot_results raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="deprecated"):
            plot_results()


class TestRealIntegration:
    """INTEGRATION TESTS: Real data visualization."""

    def test_plot_real_coral_images(self):
        """INTEGRATION: Plot real coral images from dataset."""
        from utils.data_loader import CoralDataset
        from utils.preprocessing import get_transforms
        import yaml

        # Load config
        with open('configs/config.yaml') as f:
            config = yaml.safe_load(f)

        # Load dataset with transforms
        transforms = get_transforms(config, 'val')
        dataset = CoralDataset('data/splits/val.csv', transform=transforms)

        # Get a batch of real images
        images = []
        labels = []
        for i in range(min(9, len(dataset))):
            img, label = dataset[i]
            images.append(img)
            labels.append(label)

        images = torch.stack(images)
        labels = torch.tensor(labels)

        # Plot real images
        fig = plot_sample_grid(
            images,
            labels,
            class_names=['bleached', 'healthy'],
            num_samples=9
        )

        assert isinstance(fig, plt.Figure)
        # Verify we have images displayed
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_confusion_matrix_from_real_model(self):
        """INTEGRATION: Plot confusion matrix from real predictions."""
        from utils.metrics import compute_confusion_matrix

        # Simulate realistic predictions
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = y_true.copy()
        # Add 20% error
        errors = np.random.choice(100, 20, replace=False)
        y_pred[errors] = 1 - y_pred[errors]

        # Compute real confusion matrix
        cm = compute_confusion_matrix(y_true, y_pred)

        # Plot it
        fig = plot_confusion_matrix(cm, ['bleached', 'healthy'])

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_training_curves_realistic_data(self):
        """INTEGRATION: Plot realistic training curves."""
        # Simulate realistic training
        epochs = 20
        train_losses = [0.8 * (0.95 ** i) for i in range(epochs)]
        val_losses = [0.9 * (0.94 ** i) for i in range(epochs)]
        train_accs = [0.5 + 0.4 * (1 - 0.95 ** i) for i in range(epochs)]
        val_accs = [0.5 + 0.35 * (1 - 0.94 ** i) for i in range(epochs)]

        fig = plot_training_curves(
            train_losses,
            val_losses,
            train_accs,
            val_accs
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_denormalize_real_coral_image(self):
        """INTEGRATION: Test denormalization on real coral image."""
        from utils.data_loader import CoralDataset
        from utils.preprocessing import get_transforms
        import yaml

        # Load one real coral image with transforms
        with open('configs/config.yaml') as f:
            config = yaml.safe_load(f)

        transforms = get_transforms(config, 'val')
        dataset = CoralDataset('data/splits/val.csv', transform=transforms)

        img_tensor, label = dataset[0]

        # Denormalize
        img_denorm = denormalize_image(img_tensor)

        # Verify result is reasonable
        assert img_denorm.shape == (224, 224, 3)
        assert img_denorm.min() >= 0.0
        assert img_denorm.max() <= 1.0
        # Should have variation (not all zeros or ones)
        assert img_denorm.std() > 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
